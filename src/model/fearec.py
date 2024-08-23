import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FEARecBlock

"""
[Paper]
Author: Xinyu Du et al.
Title: "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation."
Conference: SIGIR 2023

[Code Reference]
https://github.com/sudaada/FEARec
"""

class FEARecEncoder(nn.Module):
    def __init__(self, args):
        super(FEARecEncoder, self).__init__()
        self.args = args

        self.blocks = []
        for i in range(args.num_hidden_layers):
            self.blocks.append(FEARecBlock(args, layer_num=i))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):

        all_encoder_layers = [ hidden_states ]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])

        return all_encoder_layers

class FEARecModel(SequentialRecModel):
    def __init__(self, args):
        super(FEARecModel, self).__init__(args)
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = FEARecEncoder(args)
        self.batch_size = args.batch_size
        self.gamma = 1e-10

        # arguments for FEARec
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.tau = args.tau
        self.fredom = eval(args.fredom)
        self.fredom_type = args.fredom_type
        self.ssl = args.ssl
        self.sim = args.sim
        self.lmd_sem = args.lmd_sem
        self.lmd = args.lmd

        self.apply(self.init_weights)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = z[:, -1, :]
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]

        # cross-entropy loss
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        # Unsupervised NCE: original vs dropout
        if self.ssl in ['us', 'un']:
            aug_seq_output = self.forward(input_ids)
            nce_logits, nce_labels = self.info_nce(seq_output, aug_seq_output, temp=self.tau,
                                                batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised NCE: original vs semantic augmentation
        if self.ssl in ['us', 'su']:
            sem_aug = same_target
            sem_aug_seq_output = self.forward(sem_aug)
            sem_nce_logits, sem_nce_labels = self.info_nce(seq_output, sem_aug_seq_output, temp=self.tau,
                                                        batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
        
        # Unsupervised + Supervised NCE: dropout vs semantic augmentation
        if self.ssl == 'us_x':
            # unsupervised
            aug_seq_output = self.forward(input_ids)
            # supervised
            sem_aug = same_target
            sem_aug_seq_output = self.forward(sem_aug)

            sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                        batch_size=input_ids.shape[0], sim=self.sim)

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
            
        if self.fredom:
            seq_output_f = torch.fft.rfft(seq_output, dim=1, norm='ortho')
            aug_seq_output_f = torch.fft.rfft(aug_seq_output, dim=1, norm='ortho')
            sem_aug_seq_output_f = torch.fft.rfft(sem_aug_seq_output, dim=1, norm='ortho')

            if self.fredom_type in ['us', 'un']:
                loss += 0.1 * abs(seq_output_f - aug_seq_output_f).flatten().mean()

            if self.fredom_type in ['us', 'su']:
                loss += 0.1 * abs(seq_output_f - sem_aug_seq_output_f).flatten().mean()
                
            if self.fredom_type == 'us_x':
                loss += 0.1 * abs(aug_seq_output_f - sem_aug_seq_output_f).flatten().mean()

        return loss

