import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer_rec
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from obs_dataset import get_seq_dic, get_dataloder, get_rating_matrix
import interv_dataset 
from sklearn.metrics import roc_auc_score

#get_seq_dic, get_dataloder, generation_matrix_flex


def interv_metric(prob_list_interv, pred_list_interv, answer_list_interv,logger):
    count1 = 0.9
    count2 = 0.8
    count3 = 0.5

    pred_list_interv_convert_1 = np.piecewise(prob_list_interv, [prob_list_interv < count1, prob_list_interv >= count1], [0, 1])
    pred_list_interv_convert_2 = np.piecewise(prob_list_interv, [prob_list_interv < count2, prob_list_interv >= count2], [0, 1])
    pred_list_interv_convert_3 = np.piecewise(prob_list_interv, [prob_list_interv < count3, prob_list_interv >= count3], [0, 1])

    equal1 = np.argwhere(np.equal(pred_list_interv_convert_1, answer_list_interv)==True)
    equal2 = np.argwhere(np.equal(pred_list_interv_convert_2, answer_list_interv)==True)
    equal3 = np.argwhere(np.equal(pred_list_interv_convert_3, answer_list_interv)==True)

    interv_err = torch.nn.BCELoss()(torch.from_numpy(prob_list_interv).to(torch.float32), 
                                     torch.from_numpy(answer_list_interv).to(torch.float32))

    interv_err = torch.nn.BCELoss()(torch.from_numpy(pred_list_interv).to(torch.float32), torch.from_numpy(answer_list_interv).to(torch.float32))
    equal = np.argwhere(np.equal(pred_list_interv, answer_list_interv)==True)

    logger.info('interventional error, BCE loss: {}, accuracy percentage @ 0.1: {}, @ 0.2: {}, @ 0.5: {}, AUC: {}'.format(interv_err, 
                len(equal1)/(len(pred_list_interv)*len(pred_list_interv[0])), 
                len(equal2)/(len(pred_list_interv)*len(pred_list_interv[0])),
                len(equal3)/(len(pred_list_interv)*len(pred_list_interv[0])),
                roc_auc_score(answer_list_interv, prob_list_interv)))

    return [-interv_err, len(equal)/(len(pred_list_interv)*len(pred_list_interv[0]))]

def main():

    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = interv_dataset.get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1
    interv_train_dataloader, interv_eval_dataloader, interv_test_dataloader = interv_dataset.get_dataloder(args,seq_dic)

    obs_seq_dic, obs_max_item, num_users = get_seq_dic(args)
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,obs_seq_dic)
    
    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.obs_data_name+'_same_target.npy')
    

    logger.info(str(args))
    # define model here
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)
    trainer = Trainer_rec(model, train_dataloader, eval_dataloader, test_dataloader, interv_test_dataloader, args, logger)
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.obs_data_name, seq_dic, max_item)
    args.test_obs_matrix = interv_dataset.generation_matrix_flex(seq_dic['user_seq'], seq_dic['num_users'], max_item+1, pred_step=1)
    args.test_interv_matrix =  interv_dataset.generation_matrix_flex(seq_dic['rec_seq'], seq_dic['num_users'], max_item+1, pred_step=3)

    # if args.do_eval:
    #     args.test_obs_matrix = interv_dataset.generation_matrix_flex(seq_dic['user_seq'], seq_dic['num_users'], max_item+1, pred_step=1)
    #     args.test_interv_matrix =  interv_dataset.generation_matrix_flex(seq_dic['rec_seq'], seq_dic['num_users'], max_item+1, pred_step=3) 
        

# evaluate pre-trained model.
    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            logger.info(f"Load model from {args.checkpoint_path} for test!")
            interv_test_score, scores, prob_list_interv, pred_list_interv, answer_list_interv = trainer.test(0,seq_dic['num_users'], ratio = 0.2)

# train model
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        # Early stops the training if validation loss doesn't improve after a given patience.
        for epoch in range(args.epochs):
            trainer.train(epoch, args.TE_model)
            interv_test_score, scores, prob_list_interv, pred_list_interv, answer_list_interv = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        interv_test_score, test_score, prob_list_interv, pred_list_interv, answer_list_interv = trainer.test(0, seq_dic['num_users'], ratio = 0.5)


    interv_err = interv_metric(prob_list_interv, pred_list_interv, answer_list_interv,logger)

    logger.info(args.train_name)

main()
