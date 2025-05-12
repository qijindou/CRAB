#!/usr/bin/env python
#coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import os
import time
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

import util
from util import ModeType
from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
from evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from model.classification.textcnn import TextCNN
from model.classification.textrnn import TextRNN
from model.loss import ClassificationLoss
from model.model_util import get_optimizer
from queryStrategy import rand_select, get_query_index, query_score


def get_dataset(conf, args):
    train_dataset = ClassificationDataset(
        conf, args, conf.data.train_json_files, generate_dict=True)
    collate_fn = ClassificationCollator(conf, len(train_dataset.label_map), args)

    vali_dataset = Subset(train_dataset, list(range(len(train_dataset)-1000, len(train_dataset))))
    vali_data_loader = DataLoader(
        vali_dataset, batch_size=conf.train.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    test_dataset = ClassificationDataset(conf, args, conf.data.test_json_files)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_dataset, collate_fn, vali_data_loader, test_data_loader


def get_al_dataloader(conf, train_dataset, collate_fn, sample_ls, train_unlabel_ls, AL_last):
    """Get data loader: Train, Validate, Test
    """
    train_label_subset_dataset = Subset(train_dataset, sample_ls)
    train_unlabel_subset_dataset = Subset(
        train_dataset, train_unlabel_ls) if len(train_unlabel_ls) != 0 else None

    train_al_dataloader = DataLoader(
        train_label_subset_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)
    train_unlabel_dataloader = DataLoader(
        train_unlabel_subset_dataset, batch_size=conf.train.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True) if len(train_unlabel_ls) != 0 else None
    if len(train_unlabel_ls) == 0:
        AL_last = True

    return train_al_dataloader, train_unlabel_dataloader, AL_last


def get_classification_model(model_name, dataset, conf, args):
    """Get classification model from configuration
    """
    model = globals()[model_name](dataset, conf, args)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn

    def train(self, data_loader, model, optimizer, stage, epoch, i_al, n_train=0, n_ensemble=0):
        model.update_lr(optimizer, epoch)
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch, i_al, n_train=n_train, n_ensemble=n_ensemble, mode=ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch, i_al, n_train=0, n_ensemble=0, query=False):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch, i_al, n_train=n_train, n_ensemble=n_ensemble, mode=ModeType.EVAL, query=query)

    def run(self, data_loader, model, optimizer, stage,
            epoch, i_al, n_train=0, n_ensemble=0, mode=ModeType.EVAL, query=False):
        if query:
            predict_probs = []
            for batch in data_loader:
                logits = model(batch)
                result = torch.sigmoid(logits).cpu().tolist()
                predict_probs.extend(result)
            
            return predict_probs

        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            logits = model(batch) 
            loss = self.loss_fn(
                logits,
                batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                False)
            if mode == ModeType.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
            total_loss += loss.item()
            result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
            standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])
        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k, is_flat=self.conf.eval.is_flat, 
                    epoch=epoch, i_al=i_al, n_train=n_train, n_ensemble=n_ensemble)
            self.logger.warn(
                "%s performance at round %d epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % (
                    stage, i_al, epoch, precision_list[0][cEvaluator.MICRO_AVERAGE],
                    recall_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MACRO_AVERAGE],
                    right_list[0][cEvaluator.MICRO_AVERAGE],
                    predict_list[0][cEvaluator.MICRO_AVERAGE],
                        standard_list[0][cEvaluator.MICRO_AVERAGE], total_loss))
            return fscore_list[0][cEvaluator.MICRO_AVERAGE]


def get_ensemble_prediction(conf, args, n_train, model, optimizer, train_unlabel_dataloader):
    ensemble_predicts_ls = []
    for n_ensemble in range(1, 6):
        model_file_prefix = f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/{args.model}_"
        load_checkpoint(f"{model_file_prefix}train_best", model, optimizer)
        
        predict_probs = []
        for batch in train_unlabel_dataloader:
            logits = model(batch)
            result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
        ensemble_predicts_ls.append(predict_probs)

    ensemble_predicts_ls = torch.tensor(ensemble_predicts_ls, device="cpu")
    E, X, Y = ensemble_predicts_ls.shape
    z = ensemble_predicts_ls.transpose(0, 1).transpose(1, 2).reshape((X*Y, E)).unsqueeze(-1)
    prob_X_E_Y = torch.clamp(torch.cat((z, 1-z), dim=-1), min=1e-7).to("cpu")
    Py_X_n = torch.mean(ensemble_predicts_ls, dim=0).to(torch.float32).to("cpu")
    return prob_X_E_Y, Py_X_n


def load_checkpoint(file_name, model, optimizer):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def save_checkpoint(state, file_prefix):
    torch.save(state, f"{file_prefix}{state['ckp_status']}_best")


def train(conf, args):
    if not os.path.exists(f"{args.output_dir}checkpoint"): os.makedirs(f"{args.output_dir}checkpoint")
    logger = util.Logger(conf, args)

    model_name = args.model
    train_dataset, collate_fn, vali_data_loader, test_data_loader = get_dataset(conf, args)
    empty_dataset = ClassificationDataset(conf, args, [], mode="train")

    evaluator = cEvaluator(f"{args.output_dir}{conf.eval.dir}")
    loss_fn = ClassificationLoss(label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
    trainer = ClassificationTrainer(empty_dataset.label_map, logger, evaluator, conf, loss_fn)

    model = get_classification_model(model_name, empty_dataset, conf, args)
    optimizer = get_optimizer(conf, model)
    save_checkpoint({"ckp_status": "start",
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                    }, f"{args.output_dir}checkpoint/")

    for n_train in range(1, 6):
        sample_ls = []
        train_unlabel_ls = list(range(len(train_dataset)))
        AL_last = False
        i_al = 1
        performance_dict = {}
        sample_ls, train_unlabel_ls = rand_select(sample_ls= sample_ls, train_unlabel_ls=train_unlabel_ls, n_pick=args.n_annote, random_seed=666)

        for i_al in range(1, 1+args.num_al):
            train_al_dataloader, train_unlabel_dataloader, AL_last = \
                    get_al_dataloader(conf, train_dataset, collate_fn, sample_ls, train_unlabel_ls, AL_last)
            """
            Ensemble model
                Cold start
                Same seed for random sampling at first AL round
            """
            for n_ensemble in range(1, 6):
                logger.info(f"Check the result at: {args.output_dir}")
                logger.info(f"Sampling method: {args.sampling}; Number per round pick: {args.n_annote}; Learning rate: {conf.optimizer.learning_rate}; Empty pick: {args.empty_label}")
                logger.info(f"Random count: {n_train}; Ensemble count: {n_ensemble}; Total data: {len(train_dataset)}; Labeled data: {len(sample_ls)}; Unlabeled data: {len(train_unlabel_ls)}")
                load_checkpoint(f"{args.output_dir}checkpoint/start_best", model, optimizer)

                best_performance = 0
                if not os.path.exists(f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/"): 
                    os.makedirs(f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/")
                model_file_prefix = f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/{model_name}_"
                if n_ensemble not in performance_dict: performance_dict[n_ensemble] = []
                early_stop_ct = 0
                
                # Epoch training
                for epoch in range(1, 1+args.num_epochs):
                    start_time = time.time()
                    trainer.train(train_al_dataloader, model, optimizer, "Train", epoch, i_al, n_train, n_ensemble)
                    performance = trainer.eval(vali_data_loader, model, optimizer, "Train", epoch, i_al, n_train, n_ensemble)

                    if performance > best_performance: 
                        best_performance = performance
                        save_checkpoint({
                            "ckp_status": "train",
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }, model_file_prefix)
                        early_stop_ct = 0
                    else:
                        early_stop_ct += 1

                    time_used = time.time() - start_time
                    print(datetime.now(), end=" ")
                    logger.info("AL Round %d  Epoch %d  Cost time: %ds" % (i_al, epoch, time_used))
                    if early_stop_ct > conf.early_stop:
                        logger.warn("Trigger early stop at AL Round %d  Epoch %d!" % (i_al, epoch))
                        break

                load_checkpoint(f"{model_file_prefix}train_best", model, optimizer)
                best_test_performance = trainer.eval(test_data_loader, model, optimizer, "Test", epoch, i_al, n_train, n_ensemble)
                performance_dict[n_ensemble].append(best_test_performance)
                plt.figure()
                plt.plot(performance_dict[n_ensemble])
                plt.savefig(f"{args.output_dir}{n_train}_{n_ensemble}_{args.sampling}.png")
                    
                if AL_last:
                    logger.info("This is the last round of the Active Learning!")
                    continue
                if i_al == args.num_al:
                    continue

            result_file = open(f"{args.output_dir}results.json", "a")
            for key in performance_dict:
                result_file.write(json.dumps({f"S{n_train}_E{key}_performance": performance_dict[key]})+"\n")
            result_file.close()
            if i_al == args.num_al:
                final_result_file = open(f"{args.output_dir}select_result.json", "a")
                for key in performance_dict:
                    final_result_file.write(json.dumps({f"S{n_train}_E{key}_performance": performance_dict[key]})+"\n")
                final_result_file.close()

            if args.sampling == "random":
                sample_ls, train_unlabel_ls = rand_select(sample_ls=sample_ls, train_unlabel_ls=train_unlabel_ls, n_pick=args.n_annote, random_seed=None)
                prob_X_E_Y_ls, Py_X_n = get_ensemble_prediction(conf, args, n_train, model, optimizer, train_unlabel_dataloader)
                query_score(logger, conf, args, prob_X_E_Y_ls, Py_X_n, i_al, sample_ls, True)
            
            else:
                prob_X_E_Y_ls, Py_X_n = get_ensemble_prediction(conf, args, n_train, model, optimizer, train_unlabel_dataloader)
                query_score(logger, conf, args, prob_X_E_Y_ls, Py_X_n, i_al, sample_ls, True)
                sample_ls, train_unlabel_ls = get_query_index(logger, conf, args, prob_X_E_Y_ls, Py_X_n, i_al, sample_ls, train_unlabel_ls)
                
            sample_file = open(f"{args.output_dir}sampleLS.json", "a")
            sample_file.write(json.dumps(f"{i_al}:{sample_ls}")+"\n")
            sample_file.close()
        logger.info(f"This is the last epoch of the setting with random seed: {n_train}!")
    logger.info("End of training! Great job!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="The location of configure file")
    parser.add_argument("--output_dir", help="The location for output")
    parser.add_argument("--model", default="TextCNN", choices=["TextCNN", "TextRNN"], help="Select from TextCNN or TextRNN")
    parser.add_argument("--num_al", default=12, type=int, help="Number of AL round")
    parser.add_argument("--num_epochs", default=80, type=int, help="Number of epoch")

    parser.add_argument("--sampling", default="random", choices=["random", "crab", "besra_mse_batch", "besra_beta_batch"], 
                        help="Chose the ensemble model-based active learning strategy")
    parser.add_argument("--n_annote", default=100, type=int, help="The query size of every active learning around")
    parser.add_argument("--empty_label", default=200, type=int, help="The size of the selection of empty label")
    
    args = parser.parse_args()
    config = Config(config_file=args.conf)

    train(config, args)
