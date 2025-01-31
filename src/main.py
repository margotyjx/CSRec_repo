import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer_csrec, Trainer_rec
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from interv_dataset import get_seq_dic, get_dataloder, generation_matrix_flex
import obs_dataset
from model.csrec import CSRecModel
from sklearn.metrics import roc_auc_score


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def interv_metric(pred_list_interv, answer_list_interv,logger, ratio = 0.2):
    count1 = 0.9
    count2 = 0.8
    count3 = 0.5
    # interv_err = torch.nn.BCELoss()(torch.from_numpy(pred_list_interv).to(torch.float32), torch.from_numpy(answer_list_interv).to(torch.float32))
    pred_list_interv_convert_1 = np.piecewise(pred_list_interv, [pred_list_interv < count1, pred_list_interv >= count1], [0, 1])
    pred_list_interv_convert_2 = np.piecewise(pred_list_interv, [pred_list_interv < count2, pred_list_interv >= count2], [0, 1])
    pred_list_interv_convert_3 = np.piecewise(pred_list_interv, [pred_list_interv < count3, pred_list_interv >= count3], [0, 1])

    equal1 = np.argwhere(np.equal(pred_list_interv_convert_1, answer_list_interv)==True)
    equal2 = np.argwhere(np.equal(pred_list_interv_convert_2, answer_list_interv)==True)
    equal3 = np.argwhere(np.equal(pred_list_interv_convert_3, answer_list_interv)==True)

    interv_err = torch.nn.BCELoss()(torch.from_numpy(pred_list_interv).to(torch.float32), torch.from_numpy(answer_list_interv).to(torch.float32))
    
    logger.info('interventional error, BCE loss: {}, accuracy percentage @ 0.1: {}, @ 0.2: {}, @ 0.5: {}, AUC: {}'.format(interv_err, 
                len(equal1)/(len(pred_list_interv)*len(pred_list_interv[0])), 
                len(equal2)/(len(pred_list_interv)*len(pred_list_interv[0])),
                len(equal3)/(len(pred_list_interv)*len(pred_list_interv[0])),
                roc_auc_score(answer_list_interv, pred_list_interv)))
    
    return [len(equal2)/(len(pred_list_interv)*len(pred_list_interv[0]))]

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

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.obs_data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    logger.info(str(args))
    # define pre-trained model here, load pre-trained model parameters
    pretrained = MODEL_DICT[args.model_type.lower()](args=args)
    args.pretrain_path = os.path.join(args.output_dir, args.load_pretrain_model + '.pt')

    original_state_dict = pretrained.state_dict()
    new_dict_pretrain = torch.load(args.pretrain_path, map_location=device)
    logger.info(new_dict_pretrain.keys())
    for key in new_dict_pretrain:
        original_state_dict[key]=new_dict_pretrain[key]
    pretrained.load_state_dict(original_state_dict)

    # defined our model
    model = CSRecModel(args=args, pretrain_model=pretrained)
    logger.info(model)

    if args.TE_model:
        obs_seq_dic, obs_max_item, num_users = obs_dataset.get_seq_dic(args)
        obs_train_dataloader, obs_eval_dataloader, obs_test_dataloader =  obs_dataset.get_dataloder(args,obs_seq_dic)
        trainer = Trainer_csrec(model, obs_train_dataloader, obs_eval_dataloader, obs_test_dataloader, args, logger)
    else:
        trainer = Trainer_csrec(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    # args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.obs_data_name, seq_dic, max_item)
    args.test_obs_matrix = generation_matrix_flex(seq_dic['user_seq'], seq_dic['num_users'], max_item+1, pred_step=1)
    args.test_interv_matrix =  generation_matrix_flex(seq_dic['rec_seq'], seq_dic['num_users'], max_item+1, pred_step=1)

# evaluate succesfully trained model.
    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            logger.info(f"Load model from {args.checkpoint_path} for test!")
            interv_test_score, test_score, pred_list_interv, answer_list_interv = trainer.test(0, seq_dic['num_users'])
# train model
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch, seq_dic['num_users'], args.TE_model)
            interv_test_score, test_score, pred_list_interv, answer_list_interv = trainer.valid(0)
            if args.TE_model:
                early_stopping(np.array(test_score[-1:]), trainer.model)
            else:
                interv_err = interv_metric(pred_list_interv, answer_list_interv,logger)
                early_stopping(np.array(interv_err), trainer.model)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        print('args checkpoint_path: ', args.checkpoint_path)

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        interv_test_score, test_score, pred_list_interv, answer_list_interv = trainer.test(0, seq_dic['num_users'])

    if args.TE_model:
        pass
    else:
        interv_err = interv_metric(pred_list_interv, answer_list_interv,logger, ratio = 0.2)

    logger.info(args.train_name)


main()
