import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

class RecDataset_interv(Dataset):
    def __init__(self, args, user_seq, rec_seq, dec_seq, pred_step = 1, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.rec_seq = []
        self.dec_seq = []
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.rec_seq_quest = []
        self.data_type = data_type
        self.pred_step = pred_step

        if self.data_type=='train':
            # goal is to predict decision on the last recommended item
            for user, rec_ids in enumerate(rec_seq):
                input_ids = rec_ids[ -(self.max_len + 2* pred_step):-2*pred_step]
                for i in range(len(input_ids)):
                    self.rec_seq.append(input_ids[:i + 1])
                    self.user_seq.append(user_seq[user])
                    self.user_ids.append(user)
            
            for dec_ids in dec_seq:
                input_ids = dec_ids[-(self.max_len + 2* pred_step):-2*pred_step]
                for i in range(len(input_ids)):
                    self.dec_seq.append(input_ids[:i + 1])

        elif self.data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:])
            for rec in rec_seq:
                self.rec_seq.append(rec[:-2*pred_step])
                self.rec_seq_quest.append(rec[-2*pred_step:-pred_step])
            for dec in dec_seq:
                self.dec_seq.append(dec[:-pred_step])
        else:
            for sequence in user_seq:
                self.user_seq.append(sequence[:])
                # observational data
            for rec in rec_seq:
                self.rec_seq.append(rec[:-pred_step])
                self.rec_seq_quest.append(rec[-pred_step:])
            for dec in dec_seq:
                self.dec_seq.append(dec[:])

        self.test_neg_items = test_neg_items

# not updated yet
    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]
        
        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index

    def __len__(self):
        return len(self.rec_seq)

    def __getitem__(self, index):
        seq_items = self.user_seq[index]
    
        rec_items = self.rec_seq[index]
        dec_items = self.dec_seq[index]
        dec_inputs = dec_items[:-self.pred_step]
        answer = dec_items[-self.pred_step:]

        pad_len_rec = self.max_len - len(rec_items)
        pad_len_dec = self.max_len - len(dec_inputs)
        
        dec_inputs = [0] * pad_len_dec + dec_inputs
        dec_inputs = dec_inputs[-self.max_len:]

        rec_items = [0] * pad_len_rec + rec_items
        rec_items = rec_items[-self.max_len:]

        if self.data_type == 'valid':
            seq_item = seq_items[:-2]
        elif self.data_type == 'train':
            seq_item = seq_items[:-3]
        else:
            seq_item = seq_items[:-1]

        pad_len_seq = self.max_len - len(seq_item)
        seq_item = [0] * pad_len_seq + seq_item
        seq_item = seq_item[-self.max_len:]

        assert len(rec_items) == self.max_len
        assert len(dec_inputs) == self.max_len
        assert len(seq_item) == self.max_len

        if self.data_type in ['valid', 'test']:
            rec_quest = self.rec_seq_quest[index]
            if self.data_type == 'valid':
                last_seq = seq_items[-2]
            else:
                last_seq = seq_items[-1]
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(rec_items, dtype=torch.long),
                torch.tensor(rec_quest, dtype=torch.long),
                torch.tensor(dec_inputs,dtype=torch.long),
                torch.tensor(seq_item, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(last_seq, dtype=torch.long),
            )

        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(rec_items, dtype=torch.long), # recommended items
                torch.tensor(dec_inputs,dtype=torch.long), # decisions
                torch.tensor(seq_item, dtype=torch.long), # observational input
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
            )

        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


# not updated yet
def generation_matrix_flex(user_seq, num_users, num_items, pred_step = 1):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-pred_step]: 
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    gen_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return gen_matrix

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, max_item, num_users

def get_rec_seqs(rec_data_file, dec_data_file):
    # would expect that the index of rec datafile and dec datafile the same as seq datafile
    rec_lines = open(rec_data_file).readlines()
    dec_lines = open(dec_data_file).readlines()
    rec_seq = []
    dec_seq = []
    item_set = set()
    assert len(rec_lines) == len(dec_lines)
    for line in rec_lines:
        user, recitems = line.strip().split(' ', 1)
        recitems = recitems.split(' ')
        recitems = [int(item) for item in recitems]
        rec_seq.append(recitems)
        rec_item_set = item_set | set(recitems)

    for line in dec_lines:
        user, decitems = line.strip().split(' ', 1)
        decitems = decitems.split(' ')
        decitems = [int(item) for item in decitems]
        dec_seq.append(decitems)
        dec_item_set = item_set | set(decitems)

    num_users = len(rec_lines)

    return rec_seq, dec_seq, num_users

def get_seq_dic(args):
    args.obs_data_file = args.data_dir + args.obs_data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(args.obs_data_file)
    args.rec_data_file = args.data_dir + args.rec_data_name + '.txt'
    args.dec_data_file = args.data_dir + args.dec_data_name + '.txt'
    rec_seq, dec_seq, num_users_rec = get_rec_seqs(args.rec_data_file, args.dec_data_file)

    assert num_users == num_users_rec

    seq_dic = {'user_seq':user_seq, 'rec_seq':rec_seq, 'dec_seq':dec_seq,'num_users':num_users}

    return seq_dic, max_item, num_users

def get_dataloder(args,seq_dic, train = True, test = True, pred_steps = 3):
# args, user_seq, rec_seq, dec_seq
    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None

    if train:
        train_dataset = RecDataset_interv(args, seq_dic['user_seq'], seq_dic['rec_seq'], seq_dic['dec_seq'], data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

        eval_dataset = RecDataset_interv(args, seq_dic['user_seq'], seq_dic['rec_seq'], seq_dic['dec_seq'], data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    if test:
        test_dataset = RecDataset_interv(args, seq_dic['user_seq'], seq_dic['rec_seq'], seq_dic['dec_seq'], pred_step= pred_steps, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader
