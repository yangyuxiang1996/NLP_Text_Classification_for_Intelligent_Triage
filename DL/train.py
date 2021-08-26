#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-22 22:01:56
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-24 21:00:29
FilePath: /Chinese-Text-Classification/DL/train.py
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import timedelta
import time
import os
import config
from utils.dataset import MedicalData, collate_fn
from models.bilstm import LSTMModel
from models.bert_model import BertModelForMedical
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, BertTokenizer
import logging
from sklearn import metrics
from utils.adamw import AdamW
from utils.loss import FocalLoss
from utils.lr_scheduler import get_linear_schedule_with_warmup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertModelForMedical, BertTokenizer),
    'roberta': (BertConfig, BertModelForMedical, BertTokenizer),
}


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(model, train_dataset, dev_dataset, test_dataset):
    start_time = time.time()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=collate_fn)
    model.train()
    if config.use_cuda:
        model.to(config.device)
    if "bert" not in config.model_name:
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config.learning_rate)
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.02
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate, 
                          eps=config.eps)
    
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    warmup_steps = int(t_total * config.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    criterion = FocalLoss(gamma=2)
    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", config.num_train_epochs)
    logging.info("  batch size = %d", config.batch_size)
    logging.info("  Num batches = %d", config.num_train_epochs * len(train_dataloader))
    logging.info("  device: {}".format(config.device))
    
    total_batch = 0
    dev_best_acc = float('-inf')
    last_improve = 0
    flag = False

    checkpoints = [path for path in os.listdir(config.save_path) if path.startswith("checkpoint")]
    if checkpoints:
        print(checkpoints)
        checkpoints = sorted(map(lambda x: os.path.splitext(x)[0].split("_"), checkpoints), key=lambda x: float(x[2]))[-1]
        dev_best_acc = float(checkpoints[-1]) / 100
        model_path = os.path.join(config.save_path, "_".join(checkpoints)+".ckpt")
        model.load_state_dict(torch.load(model_path))
        logging.info("继续训练, {}".format(model_path))
        logging.info("最大准确率: {}".format(dev_best_acc))
        
    for epoch in range(config.num_train_epochs):
        logging.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))
        for i, batch in enumerate(train_dataloader):
            attention_mask, token_type_ids = None, None
            if "bert" in config.model_name:
                token_ids, attention_mask, token_type_ids, labels, tokens = batch
            else:
                token_ids, labels, tokens = batch
            if i < 1:
                logging.info("tokens: {}\n ".format(tokens))
                logging.info("token_ids: {}\n ".format(token_ids))
                logging.info("token_ids shape: {}\n ".format(token_ids.shape))
                logging.info("attention_mask: {}\n".format(attention_mask))
                logging.info("token_type_ids: {}\n".format(token_type_ids))
                logging.info("labels: {}\n".format(labels))
                logging.info("labels shape: {}\n".format(labels.shape))
            if config.use_cuda:
                token_ids = token_ids.to(config.device)
                attention_mask = attention_mask.to(config.device)
                token_type_ids = token_type_ids.to(config.device)
                labels = labels.to(config.device)
            outputs = model((token_ids, attention_mask, token_type_ids))
            # loss = F.cross_entropy(outputs, labels)
            loss = criterion(outputs, labels)
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            
            total_batch += 1
            if total_batch % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                loss.backward()
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
            
            if total_batch % config.logging_step == 0:
                true = labels.data.cpu()
                predicts = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predicts)
                output = evaluate(model, dev_dataset)
                dev_acc, dev_loss = output[0], output[1]
                if dev_acc > dev_best_acc:
                    logging.info("saving model..........")
                    torch.save(model.state_dict(), os.path.join(config.save_path, "checkpoint_{}_{:.2f}.ckpt".format(total_batch, dev_acc*100)))
                    improve = '*'
                    last_improve = total_batch
                    dev_best_acc = dev_acc
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter:{0}, Train Loss:{1:.4f}, Train Acc:{2:.2f}%, Val Loss:{3:.4f}, Val Acc:{4:.2f}%, Time:{5} {6}'
                logging.info(msg.format(total_batch, loss.item(), train_acc*100, dev_loss,
                               dev_acc*100, time_dif, improve))
            if total_batch - last_improve > config.patient:
                logging.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    torch.save(model.state_dict(), os.path.join(config.save_path, "model.ckpt"))
    if config.predict:
        predict(model, test_dataset)

def evaluate(model, dev_dataset, print_report=True):
    logging.info("evaluate....................")
    model.eval()
    criterion = FocalLoss(gamma=2)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                sampler=dev_sampler,
                                collate_fn=collate_fn)
    dev_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            attention_mask, token_type_ids = None, None
            if "bert" in config.model_name:
                token_ids, attention_mask, token_type_ids, labels, tokens = batch
            else:
                token_ids, labels, tokens = batch
            if config.use_cuda:
                token_ids = token_ids.to(config.device)
                attention_mask = attention_mask.to(config.device)
                token_type_ids = token_type_ids.to(config.device)
                labels = labels.to(config.device)
            outputs = model((token_ids, attention_mask, token_type_ids))
            # loss = F.cross_entropy(outputs, labels)
            loss = criterion(outputs, labels)
            dev_loss += loss
            labels = labels.data.cpu().numpy()
            predicts = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predicts)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if print_report:
        report = metrics.classification_report(labels_all,
                                               predict_all,
                                               target_names=config.label2id.keys(),
                                               digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        logging.info("result: \n%s" % report)
        logging.info("Confusion matrix: \n%s" % confusion)
        return acc, dev_loss / len(dev_dataloader), report, confusion
    return acc, dev_loss / len(dev_dataloader)
    
def predict(model, test_dataset):
    # test
    logging.info("predict....................")
    model.load_state_dict(torch.load(os.path.join(config.save_path, "model.ckpt")))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, _, _ = \
        evaluate(model, test_dataset, print_report=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    logging.info(msg.format(test_loss, test_acc))
    time_dif = get_time_dif(start_time)
    logging.info("Time usage:{}".format(time_dif))
 
    
def main():
    logging.info("加载数据......")
    train_dataset = MedicalData(config.train_data_file, 
                                config.max_seq_length,
                                char_level=config.char_level)
    dev_dataset = MedicalData(config.eval_data_file,
                              config.max_seq_length,
                              char_level=config.char_level,
                              dictionary=train_dataset.tokenizer)
    test_dataset = MedicalData(config.test_data_file,
                               config.max_seq_length,
                               char_level=config.char_level,
                               dictionary=train_dataset.tokenizer)
    if config.model_name == "bilstm":
        tokenizer = train_dataset.tokenizer
        vocab_size = tokenizer.vocabulary_size
        embeddings = torch.from_numpy(train_dataset.embeddings).to(torch.float32)
        model = LSTMModel(vocab_size=vocab_size,
                          embedding_dim=config.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_classes=config.num_classes, 
                          num_layers=config.num_layers,
                          bidirectional=config.bidirectional,
                          embeddings=embeddings
                          )
    elif "bert" in config.model_name:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model_name]
        bert_config = config_class.from_pretrained(config.bert_path,
                                                   num_labels=config.num_classes,
                                                   hidden_dropout_prob=config.dropout,
                                                   cache_dir=config.cache_dir if config.cache_dir else None,)
        tokenizer = tokenizer_class.from_pretrained(config.bert_path,
                                                    do_lower_case=config.do_lower_case,
                                                    cache_dir=config.cache_dir if config.cache_dir else None,)
        model = model_class.from_pretrained(config.bert_path,
                                            config=bert_config,
                                            cache_dir=config.cache_dir if config.cache_dir else None,)
        train_dataset.tokenizer = tokenizer
        dev_dataset.tokenizer = tokenizer
        test_dataset.tokenizer = tokenizer
    
    if config.model_name != 'transformer':
        init_network(model)
    # logging.info(model.parameters)
    train(model, train_dataset, dev_dataset, test_dataset)


if __name__ == "__main__":
    main()

        

    
    
    
