import json
import pandas as pd
from labelAnalysis import read_sort, get_co_occur_matrix, label_map

def get_label_dict(conf, sample_ls):
    data_path = conf.data.train_json_files[0]
    _, dict_list = read_sort(data_path)
    update_dict_list = [dict_list[i] for i in sample_ls]
    return update_dict_list


def get_label_list(conf, args):
    label_path = f"{args.output_dir}{conf.data.dict_dir}/doc_label.dict"
    label_list = label_map(label_path)
    return label_list


def get_label_list_bert(conf, args):
    f_label_path = open(f"{args.output_dir}{conf.data.dict_dir}/labelmap", "r")
    label_map = list(json.loads(f_label_path.readline()).keys())
    f_label_path.close()
    return label_map


def co_per_max(label_list, dict_list):
    _, co_occur_set_matrix_bi = get_co_occur_matrix(label_list, dict_list)
    df_per_max = pd.DataFrame(co_occur_set_matrix_bi, columns=label_list, index=label_list)

    for i in range(len(df_per_max)):
        df_per_max.iloc[i] = df_per_max.iloc[i] / max(df_per_max.iloc[i])
    for i in range(len(df_per_max)):
        df_per_max.iloc[i,i]=1
    df_per_max[df_per_max.isna()] = 0
    return df_per_max


def co_per_max_norm(label_list, dict_list):
    _, co_occur_set_matrix_bi = get_co_occur_matrix(label_list, dict_list)
    df_per_max = pd.DataFrame(co_occur_set_matrix_bi, columns=label_list, index=label_list)

    for i in range(len(df_per_max)):
        df_per_max.iloc[i] = df_per_max.iloc[i] / max(df_per_max.iloc[i])
    
    df_wo_self = df_per_max.copy()
    for i in range(len(df_per_max)):
        df_wo_self.iloc[i,i]=0
    
    df_per_max = df_per_max / (max(df_wo_self.sum(axis=1))*2)
    
    for i in range(len(df_per_max)):
        df_per_max.iloc[i,i]=1
    df_per_max[df_per_max.isna()] = 0
    return df_per_max
