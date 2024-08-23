import os
import torch
import numpy as np
import tqdm

from model import MODEL_DICT
from trainers import Trainer_csrec, Trainer_rec
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
# from obs_dataset import get_seq_dic, get_dataloder, get_rating_matrix
import obs_dataset
import interv_dataset 
from model.csrec import CSRecModel

#get_seq_dic, get_dataloder, generation_matrix_flex


def interv_metric(pred_list_interv, answer_list_interv,logger):
    interv_err = torch.nn.BCELoss()(torch.from_numpy(pred_list_interv).to(torch.float32), torch.from_numpy(answer_list_interv).to(torch.float32))
    equal = np.argwhere(np.equal(pred_list_interv, answer_list_interv)==True)

    logger.info('interventional error, BCE loss: {}, accuracy percentage: {}'.format(interv_err, len(equal)/(len(pred_list_interv)*len(pred_list_interv[0]))))

    return [-interv_err, len(equal)/(len(pred_list_interv)*len(pred_list_interv[0]))]

def load(model, file_name, device):
    original_state_dict = model.state_dict()
    new_dict = torch.load(file_name, map_location=device)
    for key in new_dict:
        original_state_dict[key]=new_dict[key]
    model.load_state_dict(original_state_dict)
    return model

def main():

    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if args.cuda_condition else "cpu")

    ## load data

    seq_dic, max_item, num_users = interv_dataset.get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.obs_data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = interv_dataset.get_dataloder(args,seq_dic)

    logger.info(str(args))
    # define pre-trained model here, load pre-trained model parameters
    pretrained = MODEL_DICT[args.model_type.lower()](args=args)
    args.pretrain_path = os.path.join(args.output_dir, args.load_pretrain_model + '.pt')

    original_state_dict = pretrained.state_dict()
    new_dict_pretrain = torch.load(args.pretrain_path, map_location=device)
    for key in new_dict_pretrain:
        original_state_dict[key]=new_dict_pretrain[key]
    pretrained.load_state_dict(original_state_dict)

    # defined our model
    model = CSRecModel(args=args, pretrain_model=pretrained)
    args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
    model = load(model, args.checkpoint_path, device)
    model = model.to(device)

    obs_seq_dic, obs_max_item, num_users = obs_dataset.get_seq_dic(args)
    obs_train_dataloader, obs_eval_dataloader, obs_test_dataloader =  obs_dataset.get_dataloder(args,obs_seq_dic)

    args.test_obs_matrix = interv_dataset.generation_matrix_flex(seq_dic['user_seq'], seq_dic['num_users'], max_item+1, pred_step=1)
    args.test_interv_matrix =  interv_dataset.generation_matrix_flex(seq_dic['rec_seq'], seq_dic['num_users'], max_item+1, pred_step=3)

    if args.CSRec_TE_model:
        obs_model = CSRecModel(args=args, pretrain_model=pretrained)
    else:
        obs_model = MODEL_DICT[args.model_type.lower()](args=args)

    checkpoint_path = os.path.join(args.output_dir, args.load_model_TE + '.pt')
    obs_model = load(obs_model, checkpoint_path, device)
    obs_model = obs_model.to(device)
    
    # evaluation of treatment effect 
    test_data_iter = tqdm.tqdm(enumerate(test_dataloader),
                                  desc="Mode_%s:%d" % ("treatment effect", 0),
                                  total=len(test_dataloader),
                                  bar_format="{l_bar}{r_bar}")
    model.eval()
    obs_model.eval()

    pred_score_interv = None
    pred_score_obs = None

    for i, batch in test_data_iter:
        batch = tuple(t.to(device) for t in batch)
        user_ids, rec_ids, rec_quest, dec_input, user_seq, answers, last_seq = batch

        sigmoid = torch.nn.Sigmoid()

        if args.CSRec_TE_model:
            dec_input = torch.ones_like(user_seq).to(device)
            recommend_output_0 = obs_model(user_seq,user_seq, dec_input, user_ids)
        else:
            recommend_output_0 = obs_model(user_seq, user_ids)

        recommend_output_0 = recommend_output_0[:, -1, :] # recommended output

        test_item_emb_0 = obs_model.item_embeddings.weight
        rating_pred_0 = torch.matmul(recommend_output_0, test_item_emb_0.transpose(0, 1))

        prob_all_obs = sigmoid(rating_pred_0)

        if i == 0:
            pred_score_obs = prob_all_obs.cpu().detach().numpy()
        else:
            pred_score_obs = np.append(pred_score_obs, prob_all_obs.cpu().detach().numpy(), axis = 0)

# CSRec model prediction score
        recommend_output_1 = model(user_seq,rec_ids, dec_input, user_ids)
        recommend_output_1 = recommend_output_1[:, -1, :] # recommended output
        
        test_item_emb_1 = model.item_embeddings.weight
        rating_pred_1 = torch.matmul(recommend_output_1, test_item_emb_1.transpose(0, 1))
        
        prob_all_interv = sigmoid(rating_pred_1)

        if i == 0:
            pred_score_interv = prob_all_interv.cpu().detach().numpy()
        else:
            pred_score_interv = np.append(pred_score_interv, prob_all_interv.cpu().detach().numpy(), axis = 0)

    Difference = pred_score_interv - pred_score_obs

    logger.info(Difference)

main()
