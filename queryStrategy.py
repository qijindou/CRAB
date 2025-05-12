import json
import torch
import random
import cupy as cp
import numpy as np
import pandas as pd
import cupyx.scipy.special as cs

from correlation import co_per_max, co_per_max_norm, get_label_dict, get_label_list, get_label_list_bert
from labelAnalysis import get_pred_label_dict, get_co_cluster_dict_hiere, get_uncorrelated_pair_dict, get_unco_index, read_sort, get_not_co_occur_matrix, get_co_cluster_dict_hiere_ct
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from torch.nn.functional import normalize
from torch.utils.dlpack import to_dlpack, from_dlpack

def random_generator_for_x_prime(x_dim, size):
    sample_indices = random.sample(range(0, x_dim), round(x_dim * size))
    return sorted(sample_indices)

def kmeans(rr, k):

    kmeans = KMeans(n_clusters=k, n_init="auto").fit(rr)
    centers = kmeans.cluster_centers_
    # find the nearest point to centers
    centroids = cdist(centers, rr).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(rr)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis = None)
    return centroids


def clustering(rr_X_Xp, T, n_pick):
    rr_X = torch.sum(rr_X_Xp, dim=-1)
    rr_topk_X = torch.topk(rr_X, round(rr_X.shape[0] * T))
    rr_topk_X_indices = rr_topk_X.indices.cpu().detach().numpy()
    rr_X_Xp = rr_X_Xp[rr_topk_X_indices]

    rr_X_Xp = normalize(rr_X_Xp)

    rr = kmeans(rr_X_Xp, n_pick)
    rr = [rr_topk_X_indices[x] for x in rr]

    return rr

def rand_select(sample_ls, train_unlabel_ls, n_pick, random_seed):
    n_pick = len(train_unlabel_ls) if n_pick > len(train_unlabel_ls) else n_pick
    if random_seed:
        random.seed(random_seed)
    else:
        random.seed(random.random()*10000)
    sample_ls += random.sample(train_unlabel_ls, n_pick)
    train_unlabel_ls = list(set(train_unlabel_ls) - set(sample_ls))
    return sample_ls, train_unlabel_ls

def re_index(index, origin_ls):
    return [origin_ls[idx] for idx in index]

def get_select_index(logger, conf, args, prob_X_E_Y, Py_X_n, i_al, sample_ls):
    select_index = query_score(logger, conf, args, prob_X_E_Y, Py_X_n, i_al, sample_ls)
    return select_index

def update_sample_ls(sample_ls, train_unlabel_ls, select_index):

    for idx in select_index:
        sample_ls.append(train_unlabel_ls[idx]) 
    train_unlabel_ls = list(set(train_unlabel_ls) - set(sample_ls))
    return sample_ls, train_unlabel_ls

def get_query_index(logger, conf, args, prob_X_E_Y, Py_X_n, i_al, sample_ls, train_unlabel_ls):
    select_index = get_select_index(logger, conf, args, prob_X_E_Y, Py_X_n, i_al, sample_ls)
    return update_sample_ls(sample_ls, train_unlabel_ls, select_index) 


def query_score(logger, conf, args, prob_X_E_Y, Py_X_n, i_al, sample_ls, is_analysis=False):
    """
    predict: one prediction for multi-label; one line of the predictions
    Calculate the uncertainty score line by line, this function only calculate one line with no iteration
    """

    label_dict_list = get_label_dict(conf, sample_ls)

    if "bert" in args.model.lower():
        label_list = get_label_list_bert(conf, args)
    else:
        label_list = get_label_list(conf, args)

    df_co, co_att = None, None

    df_co = co_per_max(label_list, label_dict_list)
    df_co_att = co_per_max_norm(label_list, label_dict_list)
    co_att = torch.tensor(df_co_att.values, dtype=torch.float)

    if is_analysis:
        logger.info("Query start...")

        Py_X_n = Py_X_n.detach().clone()
        Py_X_n[Py_X_n>0.5] = 1
        Py_X_n[Py_X_n<=0.5] = 0

        label_dict = get_pred_label_dict(Py_X_n, label_list)

        co_cluster_dict, n_co = get_co_cluster_dict_hiere_ct(label_list, label_dict, df_co)

        file_path = conf.data.test_json_files[0]
        _, dict_list = read_sort(file_path)
        no_co_occur_set_matrix = get_not_co_occur_matrix(label_list, dict_list)
        uncorrelated_pair_dict = get_uncorrelated_pair_dict(pd.DataFrame(no_co_occur_set_matrix, columns=label_list, index=label_list), 0.98)
        unco_index = get_unco_index(label_dict, uncorrelated_pair_dict)

        n_unco = len(unco_index)
        n_empty = len(co_cluster_dict["empty"])

        with open(f'{args.output_dir}{conf.eval.dir}/analysis_ct.json', 'a') as json_file:
            json_file.write(json.dumps({"n_co": n_co, "n_unco": n_unco, "n_empty": n_empty})+"\n")
        
        return

    
    elif args.sampling == "crab":
        logger.info("Query start...")
        xp_indices = random_generator_for_x_prime(int(prob_X_E_Y.shape[0]/Py_X_n.shape[1]), 0.0939)
        sep_label_rr_X_Xp = []

        for i in range(Py_X_n.shape[1]):
            sample_i = [n*Py_X_n.shape[1]+i for n in xp_indices]
            pr_YhThetaXp_Xp_E_Yh = prob_X_E_Y[sample_i]

            split_prob_X_E_Y = prob_X_E_Y[i::Py_X_n.shape[1]].split(co_att.shape[0]*conf.n_calcu, dim=0)
            rr_X_Xp = torch.cat([bemps_corebeta_batch(conf, t, pr_YhThetaXp_Xp_E_Yh, conf.alpha, conf.beta) for i, t in enumerate(split_prob_X_E_Y)], dim=0)

            sep_label_rr_X_Xp.append(rr_X_Xp)
        rr_X_n_Xp = torch.stack(sep_label_rr_X_Xp, dim=1)
        atted_rr_X_Xp = torch.sum(torch.matmul(co_att.unsqueeze(0), rr_X_n_Xp), dim=1)

        Py_X_n = Py_X_n.detach().clone()
        Py_X_n[Py_X_n>0.5] = 1
        Py_X_n[Py_X_n<=0.5] = 0

        label_dict = get_pred_label_dict(Py_X_n, label_list)
        co_cluster_dict = get_co_cluster_dict_hiere(label_list, label_dict, df_co)

        file_path = conf.data.test_json_files[0]
        _, dict_list = read_sort(file_path)
        no_co_occur_set_matrix = get_not_co_occur_matrix(label_list, label_dict_list)
        uncorrelated_pair_dict = get_uncorrelated_pair_dict(pd.DataFrame(no_co_occur_set_matrix, columns=label_list, index=label_list),0.98)
        unco_index = get_unco_index(label_dict, uncorrelated_pair_dict)
        
        select_index = []
        if len(unco_index) != 0: 
            if len(unco_index) >= args.n_annote:
                select_index += random.sample(unco_index, int(args.n_annote))
            else:
                select_index += unco_index

        for key in list(co_cluster_dict.keys())[:-1]:
            cu_co_idx = co_cluster_dict[key]
            if len(cu_co_idx) >= args.n_annote:
                select_index += random.sample(cu_co_idx, int(args.n_annote))
            else:
                select_index += cu_co_idx

        with open(f'{args.output_dir}{conf.eval.dir}/co_index.json', 'a') as json_file:
            json_file.write(json.dumps({"no_co": len(unco_index)}))
            for key in co_cluster_dict.keys():
                json_file.write(json.dumps({key: len(co_cluster_dict[key])}))
            json_file.write("\n")
        
        select_index = list(set(select_index))
        empty_sample = int(args.empty_label * (1 - (i_al-1) / args.num_al) ** 0.5)      

        if len(co_cluster_dict["empty"]) > empty_sample:
            select_index += random.sample(co_cluster_dict["empty"], empty_sample)
        else:
            select_index += co_cluster_dict["empty"]
        
        if len(select_index) > 2*args.n_annote:
            select_index = re_index(clustering(atted_rr_X_Xp[select_index], 0.5, args.n_annote), select_index)
        else:
            select_index = re_index(clustering(atted_rr_X_Xp[select_index], 1, args.n_annote), select_index)

        logger.info("Query Ended!")
        return select_index


def bemps_corebeta_batch(conf, probs_B_K_C, pr_YhThetaXp_Xp_E_Yh, a, b):

    probs_B_K_C = probs_B_K_C.to(conf.device)
    pr_YhThetaXp_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.to(conf.device)

    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = probs_B_K_C
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / torch.clamp(sum_pr_YThetaX_X_Y_1, min=1e-7)

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)

    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

    """
    p: pr_YhThetaXp_X_Y_Xp_E_Yh
    y: pr_Yhat_X_Y_Xp_E_Yh
    """

    p = pr_YhThetaXp_X_Y_Xp_E_Yh
    y = pr_Yhat_X_Y_Xp_E_Yh

    p_cp = cp.from_dlpack(to_dlpack(p))

    L0_cp = cs.beta(a+1, b) * cs.betainc(a+1, b, p_cp)
    L1_cp = cs.beta(a, b+1) * cs.betainc(b+1, a, 1-p_cp)

    L0 = from_dlpack(L0_cp.toDlpack()).to('cuda')
    L1 = from_dlpack(L1_cp.toDlpack()).to('cuda')

    core_beta = (1-y) * L0 + y * L1
    core_beta_X_Y_Xp = torch.sum(core_beta.sum(dim=-1), dim=-1)
    core_beta_X_Xp_Y = torch.transpose(core_beta_X_Y_Xp, 1, 2)
    core_beta_Xp_X_Y = torch.transpose(core_beta_X_Xp_Y, 0, 1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_beta_Xp_X_Y

    rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)
    rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

    return rr_X_Xp.to(torch.float32).to("cpu")

