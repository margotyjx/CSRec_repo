import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def interv_metric(pred_list_interv, answer_list_interv, ratio = 0.2):
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
    
    # logger.info('interventional error, BCE loss: {}, accuracy percentage @ 0.1: {}, @ 0.2: {}, @ 0.5: {}, AUC: {}'.format(interv_err, 
    #             len(equal1)/(len(pred_list_interv)*len(pred_list_interv[0])), 
    #             len(equal2)/(len(pred_list_interv)*len(pred_list_interv[0])),
    #             len(equal3)/(len(pred_list_interv)*len(pred_list_interv[0])),
    #             roc_auc_score(answer_list_interv, pred_list_interv)))
    
    return [interv_err, len(equal1)/(len(pred_list_interv)*len(pred_list_interv[0])), 
                len(equal2)/(len(pred_list_interv)*len(pred_list_interv[0])),
                len(equal3)/(len(pred_list_interv)*len(pred_list_interv[0])),
                roc_auc_score(answer_list_interv, pred_list_interv)]