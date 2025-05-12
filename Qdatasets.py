import os
import json
from collections import Counter

import torch
from torch.utils.data import Dataset

class MLBertDataset(Dataset):
    def __init__(self, conf, args, data_dir, tokenizer, max_len=256, generate_dict=False):
        self.conf = conf
        self.args = args
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labelmap = None
        if generate_dict == True:
            self.labelmap = self._get_labelmap_()
        else:
            f_labelmap = open(f"{self.args.output_dir}/{self.conf.data.dict_dir}/labelmap")
            self.labelmap = list(json.loads(f_labelmap.readline()).keys())
            f_labelmap.close()

        self.texts, self.labels = self._data_formalize_()
        # self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }
    
    def _get_labelmap_(self):
        data_file = open(self.conf.data.train_json_files[0], "r")
        label_ls = []

        for line in data_file.readlines():
            data_dict = json.loads(line)
            label_ls += data_dict["doc_label"]
        data_file.close()

        label_ct = Counter(label_ls)
        if not os.path.exists(f"{self.args.output_dir}/{self.conf.data.dict_dir}"): os.makedirs(f"{self.args.output_dir}/{self.conf.data.dict_dir}")
        f_labelmap = open(f"{self.args.output_dir}/{self.conf.data.dict_dir}/labelmap", "w")
        f_labelmap.write(json.dumps(label_ct))
        f_labelmap.close()

        return {key: idx for idx, key in enumerate(label_ct)}
    
    def _data_formalize_(self):
        file = open(self.data_dir, "r")
        texts = []
        labels = []

        for line in file.readlines():
            data_dict = json.loads(line)
            doc_label = data_dict["doc_label"]
            texts.append(' '.join(data_dict["doc_token"]))
            labels.append([1 if label in doc_label else 0 for label in self.labelmap])
        file.close()
        
        return texts, labels


