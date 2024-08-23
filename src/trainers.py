import tqdm
import torch
import numpy as np

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k

class Trainer_rec:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader,interv_dataloader, args, logger):
        super(Trainer_rec, self).__init__()

        self.args = args
        self.num_users = args.num_users
        self.item_size = args.item_size
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.interv_dataloader = interv_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")

    def train(self, epoch, TE_model = False):
        self.iteration(epoch, self.train_dataloader, self.num_users, self.interv_dataloader, TE_model, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, self.num_users, self.interv_dataloader, TE_model = False, train=False)

    def test(self, epoch, num_users):
        self.args.train_matrix = self.args.test_obs_matrix
        return self.iteration(epoch, self.test_dataloader, num_users, self.interv_dataloader, TE_model = False, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name, map_location=self.device)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def iteration(self, epoch, dataloader, num_users, interv_test_dataloader, TE_model = False, train=True, obs_test = True, interv_test = True):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        # rec_data_iter = enumerate(dataloader)
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                user_ids, input_ids, answers, neg_answer, same_target = batch
                if self.args.TE_model:
                    loss = self.model.calculate_loss_treatment(input_ids, answers, neg_answer, same_target, user_ids)
                else:
                    loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)
                    
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            # test or validation
            self.model.eval()
            pred_list_obs = None
            answer_list_obs = None
            above_threshold = 0.

            pred_list_interv = None
            answer_list_interv = None

            interv_data_iter = enumerate(interv_test_dataloader)

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                # user_ids, rec_ids, rec_quest, dec_input, user_seq, answers, last_seq = batch
                user_ids, input_ids, answers, _, _ = batch
                
                recommend_output = self.model.predict(input_ids, user_ids)
                recommend_output = recommend_output[:, -1, :]# recommended
                
                rating_pred = self.predict_full(recommend_output)
                rating_pred_np = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                
                try:
                    rating_pred_np[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                except: # bert4rec
                    rating_pred_np = rating_pred_np[:, :-1]
                    rating_pred_np[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # ind: get indices for largest 20 values
                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition time complexity O(n)  argsort O(nlogn)
                # The minus sign "-" indicates a larger value.
                ind = np.argpartition(rating_pred_np, -20)[:, -20:]
                # Take the corresponding values from the corresponding dimension 
                # according to the returned subscript to get the sub-table of each row of topk
                arr_ind = rating_pred_np[np.arange(len(rating_pred_np))[:, None], ind]
                # Sort the sub-tables in order of magnitude.
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred_np)), ::-1]
                # retrieve the original subscript from index again
                batch_pred_list = ind[np.arange(len(rating_pred_np))[:, None], arr_ind_argsort]
                if i == 0:
                    pred_list_obs = batch_pred_list
                    answer_list_obs = answers.cpu().data.numpy()
                else:
                    pred_list_obs = np.append(pred_list_obs, batch_pred_list, axis=0)
                    answer_list_obs = np.append(answer_list_obs, answers.cpu().data.numpy(), axis=0)

                if obs_test == True:
                    for usr, ans in enumerate(answers.cpu().data.numpy()):
                        # hitting ratio
                        if ans in batch_pred_list[usr]:
                            above_threshold += 1
            
            # test_info = {
            # "Epoch": epoch,
            # "Above threshold": "{:.4f}".format(above_threshold/num_users)}
            # self.logger.info(test_info)

            scores, result_info = self.get_full_sort_score(epoch, answer_list_obs, pred_list_obs)
                
            if interv_test == True:
                for i, batch in interv_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, rec_ids, rec_quest, dec_input, user_seq, answers, last_seq = batch

                    # need to work on input of the model from interventional data
                    recommend_output = self.model.predict(user_seq, user_ids)
                    recommend_output = recommend_output[:, -1, :] # recommended output
                    
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred_np = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    try:
                        rating_pred_np[self.args.test_rating_matrix[batch_user_index].toarray() > 0] = 0
                    except: # bert4rec
                        rating_pred_np = rating_pred_np[:, :-1]
                        rating_pred_np[self.args.test_rating_matrix[batch_user_index].toarray() > 0] = 0
                    
                    A20 = int(self.item_size * 0.2)
                    ind = np.argpartition(rating_pred_np, -A20)[:, -A20:]
                    arr_ind = rating_pred_np[np.arange(len(rating_pred_np))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred_np)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred_np))[:, None], arr_ind_argsort]

                    pred_step = len(rec_quest[0])

                    for step in range(pred_step):
                        dec_pred = []
                        for usr, ans in enumerate(rec_quest[:,step].cpu().data.numpy()):
                            # if the recommended is in the top 20, count it as dec = 1, otherwise 0
                            # print("usr and ans: ", usr, ans, batch_pred_list.shape)
                            if ans in batch_pred_list[usr]:
                                dec_pred.append(1)
                            else:
                                dec_pred.append(0)
                        
                        if step == 0:
                            prob_rec = (np.array(dec_pred)).reshape(-1,1)
                        else:
                            prob_rec = np.append(prob_rec, (np.array(dec_pred)).reshape(-1,1), axis=1)

                    if i == 0:
                        pred_list_interv = prob_rec
                        answer_list_interv = answers.cpu().detach().numpy()
                    else:
                        pred_list_interv = np.append(pred_list_interv, prob_rec, axis = 0)
                        answer_list_interv = np.append(answer_list_interv, answers.cpu().detach().numpy(), axis = 0)

            

            return scores, pred_list_interv, answer_list_interv


class Trainer_csrec:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super(Trainer_csrec, self).__init__()

        self.args = args
        self.logger = logger
        self.num_users = args.num_users
        self.item_size = args.item_size
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")

    def train(self, epoch, num_users, TE_model):
        self.iteration(epoch, self.train_dataloader, num_users, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.test_obs_matrix
        return self.iteration(epoch, self.eval_dataloader, self.num_users, train=False, obs_test = True, 
                              dec_input_choice = None, interv_test = True)
    

    def test(self, epoch, num_users, obs_test = True, dec_input_choice = None, interv_test = True):
        self.args.train_matrix = self.args.test_obs_matrix
        return self.iteration(epoch, self.test_dataloader, num_users, train=False, obs_test = obs_test, 
                              dec_input_choice = dec_input_choice, interv_test = interv_test)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name, map_location=self.device)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def iteration(self, epoch, dataloader, num_users, train=True, obs_test = True, dec_input_choice = "rating", interv_test = True, cutoff = 0.85):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        # rec_data_iter = enumerate(dataloader)
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        
        if train:
            self.model.train()
            rec_loss = 0.0
            bce_loss = 0.0
            constraint_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                # user_ids, rec_ids, dec_input, user_seq, answers, same_target = batch
                if self.args.TE_model:
                    user_ids, user_seq, answers, neg_answer, same_target = batch
                    loss = self.model.calculate_loss_treatment(user_seq, answers, same_target, user_ids, self.device)
                    bce = loss
                    constraint = loss
                else:
                    user_ids, rec_ids, dec_input, user_seq, answers, same_target = batch
                    loss, bce, constraint = self.model.calculate_loss(rec_ids, dec_input, user_seq, answers, same_target, user_ids, self.device, alpha0 = 1., alpha1 = 1.)
                    
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()
                bce_loss += bce.item()
                constraint_loss += constraint.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
                "bce_loss": '{:.4f}'.format(bce_loss / len(rec_data_iter)),
                "constraint_loss": '{:.4f}'.format(constraint_loss / len(rec_data_iter))
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            self.model.eval()
            above_threshold = 0.

            pred_list_interv = None
            answer_list_interv = None

            pred_list_obs = None
            answer_list_obs = None

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)

                if self.args.TE_model:
                    user_ids, user_seq, last_seq, neg_answer, same_target = batch
                    interv_test = False
                else:
                    user_ids, rec_ids, rec_quest, dec_input, user_seq, answers, last_seq = batch

                if obs_test == True:
                    # test on observational data
                    if dec_input_choice == "rating":
                        recommend_output = self.model(user_seq,user_seq, dec_input, user_ids)
                    else:
                        dec_input = torch.ones_like(user_seq).to(self.device)
                        recommend_output = self.model(user_seq,user_seq, dec_input, user_ids)
                    
                    recommend_output = recommend_output[:, -1, :] # recommended output
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred_np = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    sigmoid = torch.nn.Sigmoid()
                    prob_all = sigmoid(rating_pred)

                    ind_range = (torch.arange(0,len(last_seq))).to(self.device)
                    inds = torch.concatenate((ind_range.reshape(-1,1),last_seq.reshape(-1,1)), dim=1)
                    prob_rec = prob_all[inds[:,0],inds[:,1]]

                    prob_rec = prob_rec.cpu().detach().numpy()
                    # above_threshold += len(np.argwhere(prob_rec>cutoff))

                    try:
                        rating_pred_np[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    except: # bert4rec
                        rating_pred_np = rating_pred_np[:, :-1]
                        rating_pred_np[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    # ind: get indices for largest 20 values
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition time complexity O(n)  argsort O(nlogn)
                    # The minus sign "-" indicates a larger value.
                    ind = np.argpartition(rating_pred_np, -20)[:, -20:]
                    # Take the corresponding values from the corresponding dimension 
                    # according to the returned subscript to get the sub-table of each row of topk
                    arr_ind = rating_pred_np[np.arange(len(rating_pred_np))[:, None], ind]
                    # Sort the sub-tables in order of magnitude.
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred_np)), ::-1]
                    # retrieve the original subscript from index again
                    batch_pred_list = ind[np.arange(len(rating_pred_np))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list_obs = batch_pred_list
                        answer_list_obs = last_seq.cpu().data.numpy()
                    else:
                        pred_list_obs = np.append(pred_list_obs, batch_pred_list, axis=0)
                        answer_list_obs = np.append(answer_list_obs, last_seq.cpu().data.numpy(), axis=0)
                    
                

                if interv_test == True:
                    recommend_output = self.model(user_seq,rec_ids, dec_input, user_ids)
                    recommend_output = recommend_output[:, -1, :] # recommended output
                    
                    rating_pred = self.predict_full(recommend_output)
                    sigmoid = torch.nn.Sigmoid()
                    prob_all = sigmoid(rating_pred)

                    pred_step = len(rec_quest[0])
                    ind_range = (torch.arange(0,len(last_seq))).to(self.device)


                    for step in range(pred_step):
                        inds = torch.concatenate((ind_range.reshape(-1,1),(rec_quest[:,step]).reshape(-1,1)), dim=1)
                        prob_rec_j = prob_all[inds[:,0],inds[:,1]]
                        
                        if step == 0:
                            prob_rec = prob_rec_j.cpu().detach().numpy().reshape(-1,1)
                        else:
                            prob_rec = np.append(prob_rec, prob_rec_j.cpu().detach().numpy().reshape(-1,1), axis=1)

                    if i == 0:
                        pred_list_interv = prob_rec
                        answer_list_interv = answers.cpu().detach().numpy()
                    else:
                        pred_list_interv = np.append(pred_list_interv, prob_rec, axis = 0)
                        answer_list_interv = np.append(answer_list_interv, answers.cpu().detach().numpy(), axis = 0)

            scores_obs, result_info_obs = self.get_full_sort_score(epoch, answer_list_obs, pred_list_obs)
            # test_info = {
            # "Epoch": epoch,
            # "Above threshold": "{:.4f}".format(above_threshold/num_users)}
            # self.logger.info(test_info)

            return scores_obs, pred_list_interv, answer_list_interv
