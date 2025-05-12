import json
import numpy as np
from itertools import combinations, product

def read_sort(file_path: str):
    dict_list = []

    file = open(file_path, 'r', encoding='utf-8')
    for line in file:
        line = line.strip()
        if line:  
            data_dict = json.loads(line)
            dict_list.append(data_dict)
    file.close()

    label_list = []
    for dict in dict_list:
        label_list += dict['doc_label']
    label_list = list(sorted(set(label_list)))
    return label_list, dict_list


def get_co_occur_matrix(label_list: list, dict_list: list) -> np:
    co_occur_pair_ls = []
    label_occur_ct = {c:0 for c in label_list}
    index_map = {value: idx for idx, value in enumerate(label_list)}

    for dict in dict_list:
        for c in dict['doc_label']:
            label_occur_ct[c] += 1
        if len(dict['doc_label']) == 1: continue
        co_occur_pair_ls += combinations(sorted(dict['doc_label'], key=lambda x: index_map[x]),2)
        
    co_occur_set_matrix = np.zeros((len(label_list), len(label_list)))
    co_occur_set_matrix_bi = np.zeros((len(label_list), len(label_list)))
    for pair in co_occur_pair_ls:
        co_occur_set_matrix[label_list.index(pair[0]), label_list.index(pair[1])] += 1
        co_occur_set_matrix_bi[label_list.index(pair[0]), label_list.index(pair[1])] += 1
        co_occur_set_matrix_bi[label_list.index(pair[1]), label_list.index(pair[0])] += 1
    for key in label_occur_ct:
        co_occur_set_matrix[label_list.index(key), label_list.index(key)] = label_occur_ct[key]
        co_occur_set_matrix_bi[label_list.index(key), label_list.index(key)] = label_occur_ct[key]
        
    return co_occur_set_matrix, co_occur_set_matrix_bi


def get_label_ct_dict(label_list, dict_list):
    label_ct_dict = {label:0 for label in label_list}
    for dict in dict_list:
        for label in dict['doc_label']:
            label_ct_dict[label] += 1
    return label_ct_dict


def get_not_co_occur_matrix(label_list: list, dict_list: list) -> np:
    a_nb_pair_ls = []
    not_occur_ct = {c:0 for c in label_list}

    for dict in dict_list:
        for c in dict['doc_label']:
            not_occur_ct[c] += 1
        islabel = sorted(dict['doc_label'])
        notlabel = sorted(list(set(label_list) - set(islabel)))
        a_nb_pair_ls += product(islabel, notlabel)

    a_n_b_matrix = np.zeros((len(label_list), len(label_list)))
    for pair in a_nb_pair_ls:
        a_n_b_matrix[label_list.index(pair[0]), label_list.index(pair[1])] += 1
    for key in not_occur_ct:
        a_n_b_matrix[label_list.index(key), label_list.index(key)] = not_occur_ct[key]

    return a_n_b_matrix


def label_map(file_path):
    label_file = open(file_path, 'r')
    labels = label_file.readlines()
    label_file.close()
    label_ls = [label.split('\t')[0] for label in labels]

    return label_ls


def get_pred_label_dict(Py_X_n, label_list):
    label_dict = {}
    for i in range(len(Py_X_n)):
        indices = (Py_X_n[i] == 1).nonzero(as_tuple=False)
        labels = [label_list[idx.item()] for idx in indices]
        label_dict[i] = labels
    
    return label_dict


def get_co_cluster_dict_hiere(label_list, label_dict, df_co):
    co_cluster_dict = {label: [] for label in label_list}
    co_cluster_dict["empty"] = []

    combs = list(combinations(sorted(label_list), 2))
    hirer_remove_dict = {}
    for comb in combs:
        if (df_co[comb[0]][comb[1]] <= df_co[comb[1]][comb[0]]) and df_co[comb[1]][comb[0]]>=0.7:
            hirer_remove_dict[comb] = comb[1]
        elif (df_co[comb[1]][comb[0]] <= df_co[comb[0]][comb[1]]) and df_co[comb[0]][comb[1]]>=0.7:
            hirer_remove_dict[comb] = comb[0]

    for key in label_dict.keys():
        # label_dict = label_dict_list[i]
        labels = label_dict[key]
        if len(labels) == 0:
            co_cluster_dict["empty"].append(key)
        else:
            remove_ls = []
            for comb in list(combinations(labels, 2)):
                if comb in hirer_remove_dict.keys():
                    remove_ls.append(hirer_remove_dict[comb])
            new_label_ls = list(set(labels) - set(remove_ls))
            for label in new_label_ls:
                co_cluster_dict[label].append(key)

    return co_cluster_dict


def get_co_cluster_dict_hiere_ct(label_list, label_dict, df_co):
    co_cluster_dict = {label: [] for label in label_list}
    co_cluster_dict["empty"] = []
    n_ct = 0

    combs = list(combinations(sorted(label_list), 2))
    hirer_remove_dict = {}
    for comb in combs:
        if (df_co[comb[0]][comb[1]] <= df_co[comb[1]][comb[0]]) and df_co[comb[1]][comb[0]]>=0.7:
            hirer_remove_dict[comb] = comb[1]
        elif (df_co[comb[1]][comb[0]] <= df_co[comb[0]][comb[1]]) and df_co[comb[0]][comb[1]]>=0.7:
            hirer_remove_dict[comb] = comb[0]

    for key in label_dict.keys():
        # label_dict = label_dict_list[i]
        labels = label_dict[key]
        if len(labels) == 0:
            co_cluster_dict["empty"].append(key)
        else:
            remove_ls = []
            for comb in list(combinations(labels, 2)):
                if comb in hirer_remove_dict.keys():
                    remove_ls.append(hirer_remove_dict[comb])
            new_label_ls = list(set(labels) - set(remove_ls))
            if len(new_label_ls) != len(labels): 
                n_ct += 1

            for label in new_label_ls:
                co_cluster_dict[label].append(key)

    return co_cluster_dict, n_ct


def get_uncorrelated_pair_dict(df_unco, n):
    uncorrelated_pair_dict = {}

    for i in range(len(df_unco)):
        curr_label = df_unco.columns[i]
        curr_ct = df_unco.iloc[i,i]
        if curr_ct != 0:
            chosen_labels = list(df_unco.iloc[i][(df_unco.iloc[i] / curr_ct) > n].index)
            chosen_labels.remove(curr_label)
            if chosen_labels != []:
                uncorrelated_pair_dict[curr_label] = chosen_labels
    return uncorrelated_pair_dict


def get_unco_index(label_dict, uncorrelated_pair_dict):
    unco_index = []
    for key in label_dict.keys():
        labels = label_dict[key]
        for label in labels:
            if label not in uncorrelated_pair_dict.keys():
                continue
            unco_pair = product([label], uncorrelated_pair_dict[label])
            for pair in unco_pair:
                if pair[0] in labels and pair[1] in labels:
                    unco_index.append(key)
    return list(set(unco_index))