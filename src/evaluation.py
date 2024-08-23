import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from interv_dataset import get_seq_dic, get_dataloder, get_rating_matrix
from model.csrec import CSRecModel

def load_model(model, file_name):
    original_state_dict = model.state_dict()
    new_dict = torch.load(file_name)
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

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.obs_data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    logger.info(str(args))

    #define pretrained model
    pretrained = MODEL_DICT[args.model_type.lower()](args=args)
    args.pretrain_path = os.path.join(args.output_dir, args.load_pretrain_model + '.pt')
    pretrained = load_model(pretrained, args.pretrain_path)

    # define model here
    model = CSRecModel(args=args, pretrain_model=pretrained)
    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.obs_data_name, seq_dic, max_item) 

# evaluate pre-trained model.
    if args.load_pretrain_model is None:
        logger.info(f"No model input!")
        exit(0)
    else:
        train_matrix = args.test_rating_matrix
        args.checkpoint_path = os.path.join(args.output_dir, args.load_pretrain_model + '.pt')
        model = load_model(model, args.checkpoint_path)