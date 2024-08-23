import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer_rec
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from obs_dataset import get_seq_dic, get_dataloder, get_rating_matrix
import interv_dataset 

#get_seq_dic, get_dataloder, generation_matrix_flex


def interv_metric(pred_list_interv, answer_list_interv,logger):
    interv_err = torch.nn.BCELoss()(torch.from_numpy(pred_list_interv).to(torch.float32), torch.from_numpy(answer_list_interv).to(torch.float32))
    equal = np.argwhere(np.equal(pred_list_interv, answer_list_interv)==True)

    logger.info('interventional error, BCE loss: {}, accuracy percentage: {}'.format(interv_err, len(equal)/(len(pred_list_interv)*len(pred_list_interv[0]))))

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
            scores, pred_list_interv, answer_list_interv = trainer.test(0,seq_dic['num_users'])

# train model
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        # Early stops the training if validation loss doesn't improve after a given patience.
        for epoch in range(args.epochs):
            trainer.train(epoch, args.TE_model)
            scores, pred_list_interv, answer_list_interv = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        
        trainer.save(args.checkpoint_path)

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        test_score, pred_list_interv, answer_list_interv = trainer.test(0, seq_dic['num_users'])

        # test_info = {
        #     "HR@5": '{:.4f}'.format(test_score[0]), "NDCG@5": '{:.4f}'.format(test_score[1]),
        #     "HR@10": '{:.4f}'.format(test_score[2]), "NDCG@10": '{:.4f}'.format(test_score[3]),
        #     "HR@20": '{:.4f}'.format(test_score[4]), "NDCG@20": '{:.4f}'.format(test_score[5])
        # }

    interv_err = interv_metric(pred_list_interv, answer_list_interv,logger)
    # interv_err = torch.nn.BCELoss()(torch.from_numpy(pred_list_interv).to(torch.float32), torch.from_numpy(answer_list_interv).to(torch.float32))
    # equal = np.argwhere(np.equal(pred_list_interv, answer_list_interv)==True)
    # print('interventional error, BCE loss: {}, accuracy percentage: {}'.format(interv_err, len(equal)/(len(pred_list_interv)*len(pred_list_interv[0]))))

    logger.info(args.train_name)
    # logger.info(scores)
    # logger.info(test_info)


main()
