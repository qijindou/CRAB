import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, get_linear_schedule_with_warmup

import util
from util import ModeType
from config import Config
from Qdatasets import MLBertDataset
from queryStrategy import rand_select, get_query_index

def evaluator(all_labels, all_preds):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=1)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=1)
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=1)

    label_standard = np.sum(all_labels)
    label_predict = np.sum(all_preds)
    label_right = np.sum((all_preds == 1) & (all_labels == 1))

    return f1_micro, f1_macro, precision, recall, label_standard, label_predict, label_right


def save_preds(conf, args, predicts, n_train, n_ensemble, i_al, epoch):
    eval_dir = f"{args.output_dir}{conf.eval.dir}"
    if not os.path.exists(f"{eval_dir}/probs_{n_train}/Ensemble_{n_ensemble}/"): os.makedirs(f"{eval_dir}/probs_{n_train}/Ensemble_{n_ensemble}/")
    debug_file = open(f"{eval_dir}/probs_{n_train}/Ensemble_{n_ensemble}/probs_{i_al}_{epoch}.txt", "w")
    for predict in predicts:
        prob_np = np.array(predict, dtype=np.float32)
        debug_file.write(json.dumps(prob_np.tolist())+"\n")
    return


def get_dataset(conf, args, tokenizer):
    train_dataset = MLBertDataset(
        conf, args, conf.data.train_json_files[0], tokenizer, max_len=conf.feature.max_token_len, generate_dict=True)

    vali_dataset = Subset(train_dataset, list(range(len(train_dataset)-1000, len(train_dataset))))
    vali_data_loader = DataLoader(
        vali_dataset, batch_size=conf.train.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, pin_memory=True)

    test_dataset = MLBertDataset(conf, args, conf.data.test_json_files[0], tokenizer, max_len=conf.feature.max_token_len)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, pin_memory=True)

    return train_dataset, vali_data_loader, test_data_loader


def get_al_dataloader(conf, train_dataset, sample_ls, train_unlabel_ls, AL_last):
    """Get data loader: Train, Validate, Test
    """
    train_label_subset_dataset = Subset(train_dataset, sample_ls)
    train_unlabel_subset_dataset = Subset(
        train_dataset, train_unlabel_ls) if len(train_unlabel_ls) != 0 else None

    train_al_dataloader = DataLoader(
        train_label_subset_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, pin_memory=True)
    train_unlabel_dataloader = DataLoader(
        train_unlabel_subset_dataset, batch_size=conf.train.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, pin_memory=True) if len(train_unlabel_ls) != 0 else None
    if len(train_unlabel_ls) == 0:
        AL_last = True

    return train_al_dataloader, train_unlabel_dataloader, AL_last


class ClassificationTrainer(object):
    def __init__(self, logger, conf, args):
        self.logger = logger
        self.conf = conf
        self.args = args

    def train(self, data_loader, model, optimizer, scheduler, stage, epoch, i_al, n_train=0, n_ensemble=0):
        model.train()
        return self.run(data_loader, model, optimizer, scheduler, stage, epoch, i_al, n_train=n_train, n_ensemble=n_ensemble, mode=ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, scheduler, stage, epoch, i_al, n_train=0, n_ensemble=0, query=False):
        model.eval()
        return self.run(data_loader, model, optimizer, scheduler, stage, epoch, i_al, n_train=n_train, n_ensemble=n_ensemble, mode=ModeType.EVAL, query=query)

    def run(self, data_loader, model, optimizer, scheduler, stage,
            epoch, i_al, n_train=0, n_ensemble=0, mode=ModeType.EVAL, query=False):
        if query:
            predict_probs = []
            for batch in data_loader:
                input = {k: v.to(self.conf.device) for k, v in batch.items()}
                output = model(**input)
                logits = output.logits
                result = torch.sigmoid(logits).cpu().tolist()
                predict_probs.extend(result)
            return predict_probs

        predict_probs = []
        predict_labels = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            input = {k: v.to(self.conf.device) for k, v in batch.items()}
            output = model(**input)
            logits = output.logits
            loss = output.loss
            if mode == ModeType.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                continue
            total_loss += loss.item()
            predict_probs.extend(torch.sigmoid(logits).cpu().tolist())
            predict_labels.extend((torch.sigmoid(logits) > 0.5).int().cpu().tolist())
            standard_labels.extend(input["labels"].cpu().numpy())
        if mode == ModeType.EVAL:
            save_preds(self.conf, self.args, predict_probs, n_train, n_ensemble, i_al, epoch)
            total_loss = total_loss / num_batch

            f1_micro, f1_macro, precision, recall, label_standard, label_predict, label_right = evaluator(standard_labels, predict_labels)
            self.logger.warn(
                "%s performance at round %d epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % ( 
                    stage, i_al, epoch, precision, recall, f1_micro, f1_macro, 
                    label_right, label_predict, label_standard, total_loss))
            return f1_micro


def get_ensemble_prediction(conf, args, n_train, model, optimizer, train_unlabel_dataloader):
    ensemble_predicts_ls = []

    for n_ensemble in range(1,6):
        model_file_prefix = f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/{args.model}_"
        load_checkpoint(f"{model_file_prefix}train_best", model, optimizer)
        
        predict_probs = []
        for batch in train_unlabel_dataloader:
            input = {k: v.to(conf.device) for k, v in batch.items()}
            output = model(**input)
            logits = output.logits
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
    if not os.path.exists(f"{args.output_dir}{conf.eval.dir}"):os.makedirs(f"{args.output_dir}{conf.eval.dir}")
    logger = util.Logger(conf, args)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased', force_download=None)
    train_dataset, vali_data_loader, test_data_loader = get_dataset(conf, args, tokenizer)
    model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=len(train_dataset.labelmap)).to(conf.device)
    optimizer = AdamW(model.parameters(), lr=conf.optimizer.learning_rate)
    save_checkpoint({"ckp_status": "start",
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                    }, f"{args.output_dir}checkpoint/")
    trainer = ClassificationTrainer(logger, conf, args)

    for n_train in range(1, 6):
        sample_ls = []
        train_unlabel_ls = list(range(len(train_dataset)))
        AL_last = False
        i_al = 1
        performance_dict = {}
        sample_ls, train_unlabel_ls = rand_select(sample_ls=sample_ls, train_unlabel_ls=train_unlabel_ls, n_pick=args.n_annote, random_seed=666)

        for i_al in range(1, 1+args.num_al):
            train_al_dataloader, train_unlabel_dataloader, AL_last = \
                get_al_dataloader(conf, train_dataset, sample_ls, train_unlabel_ls, AL_last)
            """
            Ensemble model
                Cold start
                Same seed for first AL round
                Random seed for else
            """

            for n_ensemble in range(1,6):
                logger.info(f"Check the result at: {args.output_dir}")
                logger.info(f"Sampling method: {args.sampling}; Number per round pick: {args.n_annote}; Learning rate: {conf.optimizer.learning_rate}; Empty pick: {args.empty_label}")
                logger.info(f"Random count: {n_train}; Ensemble count: {n_ensemble}; Total data: {len(train_dataset)}; Labeled data: {len(sample_ls)}; Unlabeled data: {len(train_unlabel_ls)}")
                load_checkpoint(f"{args.output_dir}checkpoint/start_best", model, optimizer)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_epochs*len(train_al_dataloader)*conf.optimizer.warmup_percentage, 
                                                            num_training_steps=args.num_epochs*len(train_al_dataloader))

                best_performance = 0
                if not os.path.exists(f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/"): os.makedirs(f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/")
                model_file_prefix = f"{args.output_dir}checkpoint/checkpoint_{n_train}/Ensemble_{n_ensemble}/{args.model}_"
                if n_ensemble not in performance_dict: performance_dict[n_ensemble] = []
                early_stop_ct = 0
                
                for epoch in range(1, 1+args.num_epochs):
                    start_time = time.time()
                    trainer.train(train_al_dataloader, model, optimizer, scheduler, "Train", epoch, i_al, n_train, n_ensemble)
                    performance = trainer.eval(vali_data_loader, model, optimizer, scheduler, "Train", epoch, i_al, n_train, n_ensemble)
                    if performance > best_performance: 
                        best_performance = performance
                        save_checkpoint({
                            "ckp_status": "train",
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict()
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
                best_test_performance = trainer.eval(test_data_loader, model, optimizer, scheduler, "Test", epoch, i_al, n_train, n_ensemble)
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
                sample_ls, train_unlabel_ls = rand_select(sample_ls=sample_ls, train_unlabel_ls=train_unlabel_ls, n_pick=args.n_annote, random_n_train=None)
            else:
                prob_X_E_Y_ls, Py_X_n = get_ensemble_prediction(conf, args, n_train, model, optimizer, train_unlabel_dataloader)
                sample_ls, train_unlabel_ls = get_query_index(logger, conf, args, prob_X_E_Y_ls, Py_X_n, i_al, sample_ls, train_unlabel_ls)

            sample_file = open(f"{args.output_dir}sampleLS.json", "a")
            sample_file.write(json.dumps(f"{i_al}:{sample_ls}")+"\n")
            sample_file.close()
        logger.info(f"This is the last epoch of train no: {n_train}!")
    logger.info("End of training! Great job!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="The location of configure file")
    parser.add_argument("--output_dir", help="The location for output")
    parser.add_argument("--model", default="Bert", choices=["Bert"], help="Select from TextCNN or TextRNN")
    parser.add_argument("--num_al", default=12, type=int, help="Number of AL round")
    parser.add_argument("--num_epochs", default=30, type=int, help="Number of epoch")

    parser.add_argument("--sampling", default="random", choices=["random", "crab", "besra_mse_batch", "besra_beta_batch"], 
                        help="Chose the ensemble model-based active learning strategy")
    parser.add_argument("--n_annote", default=100, type=int, help="The query size of every active learning around")
    parser.add_argument("--empty_label", default=200, type=int, help="The size of the selection of empty label")
    
    args = parser.parse_args()
    config = Config(config_file=args.conf)

    train(config, args)
