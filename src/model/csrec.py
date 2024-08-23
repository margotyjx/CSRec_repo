import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, BSARecBlock


# need to add a feed forward to predict the probability
class CSRecEncoder(nn.Module):
    def __init__(self, args):
        super(CSRecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class CSRecModel(SequentialRecModel):
    def __init__(self, args, pretrain_model):
        super(CSRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = CSRecEncoder(args)
        self.apply(self.init_weights)
        self.pretrain_model = pretrain_model

    def forward(self, user_seq, rec_ids, decision_input, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(rec_ids)
        # sequence_emb = self.add_position_decision_embedding(rec_ids, decision_input)
        sequence_emb = self.add_position_rec_decision_embedding(user_seq, rec_ids, decision_input)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )               
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output
    
    def obs_emb(self, rec_ids, decision_input, input_ids, user_ids = None, all_sequence_output=False):
        # input_ids are item ids of pure observational data
        # rec_ids and decision_input are interventional data
        # D_{t-1} = 0 --> original input, D_{t-1} = 1 --> append the last rec_ids as input
        combine_input = input_ids.clone()
        combine_input[:,:-1] = combine_input[:,1:]
        combine_input[:,-1] = rec_ids[:,-1]

        obs_dec_1 = self.pretrain_model(combine_input, user_ids, all_sequence_output)
        obs_dec_0 = self.pretrain_model(input_ids, user_ids, all_sequence_output)

        return obs_dec_0, obs_dec_1
    
    def calculate_loss(self, rec_ids, decision_input, input_ids, answers, same_target, user_ids, device, alpha0 = 1., alpha1 = 1.):
        softmax = nn.Softmax(dim=1)
        sigmoid = torch.nn.Sigmoid()
        # current time t, compute P(D_{t-1}|do(S_{t-1},S_{t-2},...,S_1),L)
        rec_ids_prev = rec_ids.clone()
        rec_ids_prev[:,2:] = rec_ids[:,:-2]
        rec_ids_prev[:,:2] = torch.zeros_like(rec_ids[:,-2:])

        dec_input_prev = decision_input.clone()
        dec_input_prev[:,1:] = decision_input[:,:-1]
        dec_input_prev[:,:1] = torch.zeros_like(decision_input[:,-1:])

        # embedding or probability in the previous step 
        emb_prev = self.forward(input_ids, rec_ids_prev, dec_input_prev, user_ids)
        emb_prev = emb_prev[:,-1,:]

        item_emb = self.item_embeddings.weight

        prob_prev_all = sigmoid(torch.matmul(emb_prev, item_emb.transpose(0, 1)))
        rec_inds_prev = rec_ids[:,-2]
        ind_range = (torch.arange(0,len(rec_inds_prev))).to(device)
        inds_prev = torch.concatenate((ind_range.reshape(-1,1),rec_inds_prev.reshape(-1,1)), dim=1)
        prob_rec_prev = prob_prev_all[inds_prev[:,0],inds_prev[:,1]]

        # current time, compute P(D_t|S_t, D_{t-1},L)
        obs_emb_t_0,obs_emb_t_1  = self.obs_emb(rec_ids, decision_input, input_ids, user_ids)
        obs_emb_t_0,obs_emb_t_1 = obs_emb_t_0[:,-1,:],obs_emb_t_1[:,-1,:]
        dec_0 = softmax(torch.matmul(obs_emb_t_0, item_emb.transpose(0, 1)))
        dec_1 = softmax(torch.matmul(obs_emb_t_1, item_emb.transpose(0, 1)))

        prob_dec_0 = dec_0[inds_prev[:,0],inds_prev[:,1]]
        prob_dec_1 = dec_1[inds_prev[:,0],inds_prev[:,1]]

        constraint_rhs = prob_dec_1*prob_rec_prev + prob_dec_0*(torch.ones_like(prob_rec_prev) - prob_rec_prev)

        # current time, compute P(D_t|do(S_{t},S_{t-1},...,S_1),L)
        rec_ids_t = rec_ids.clone()
        rec_ids_t[:,1:] = rec_ids[:,:-1]
        rec_ids_t[:,:1] = torch.zeros_like(rec_ids[:,-1:])

        emb_t = self.forward(input_ids, rec_ids_t, decision_input, user_ids)
        emb_t = emb_t[:,-1,:]
        prob_all = sigmoid(torch.matmul(emb_t, item_emb.transpose(0, 1)))

        rec_inds = rec_ids[:,-1]
        ind_range = (torch.arange(0,len(rec_inds))).to(device)
        inds = torch.concatenate((ind_range.reshape(-1,1),rec_inds.reshape(-1,1)), dim=1)
        prob_rec = prob_all[inds[:,0],inds[:,1]]

        loss = nn.BCELoss()(prob_rec.reshape(-1,1), answers.type(torch.cuda.FloatTensor))
        constraint = nn.MSELoss()(prob_rec,constraint_rhs)

        return alpha0*loss + alpha1*constraint, loss, constraint
    
    def calculate_loss_treatment(self, input_ids, answers, same_target, user_ids, device):
        softmax = nn.Softmax(dim=1)
        sigmoid = torch.nn.Sigmoid()
        
        # current time t, compute P(D_{t}|D_{t-1},L)
        item_emb = self.item_embeddings.weight
        decision_input = torch.ones_like(input_ids).to(device)
        emb_t = self.forward(input_ids, input_ids, decision_input, user_ids)
        emb_t = emb_t[:,-1,:]
        prob_all = sigmoid(torch.matmul(emb_t, item_emb.transpose(0, 1)))

        ind_range = (torch.arange(0,len(answers))).to(device)
        inds = torch.concatenate((ind_range.reshape(-1,1),answers.reshape(-1,1)), dim=1)
        prob_rec = prob_all[inds[:,0],inds[:,1]]

        loss = nn.BCELoss()(prob_rec, torch.ones_like(answers).to(answers.device).type(torch.cuda.FloatTensor))

        return loss





    # def calculate_loss(self, input_ids, decision_input, answers, neg_answers, same_target, user_ids):
    #     # probability of decision given system recommendation
    #     seq_output = self.forward(input_ids, decision_input)
    #     seq_output = seq_output[:, -1, :]
    #     item_emb = self.item_embeddings.weight
    #     logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
    #     loss = nn.CrossEntropyLoss()(logits, answers)

    #     return loss

