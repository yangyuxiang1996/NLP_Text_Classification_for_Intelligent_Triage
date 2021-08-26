#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 19:02:39
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-24 20:59:12
FilePath: /Chinese-Text-Classification/DL/config.py
Description: 
'''
import torch
import os
import numpy as np
import json

# generate config
curPath = os.path.abspath(os.path.dirname(__file__))

stopWords_file = curPath + '/data/stopwords.txt'
log_dir = curPath + '/logs/'
train_data_file = curPath + '/data/train_agg.csv'
eval_data_file = curPath + '/data/eval.csv'
test_data_file = curPath + '/data/test.csv'
label2id_file = curPath + '/data/label2id.json'
id2label_file = curPath + '/data/id2label.json'
stopwords = curPath + '/data/stopwords.txt'
vocab_path = curPath + '/data/vocab.txt'
w2v_path = curPath + '/data/w2v_agg.bin'

model_name = "roberta"
if "bert" in model_name:
    char_level = True  # bert类模型为True，其它为False
else:
    char_level = False
save_path = curPath + '/output/{}'.format(model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
use_cuda = True

device = torch.device('cuda') if use_cuda else torch.device('cpu')
label2id = json.load(open(label2id_file, "r"))
num_classes = len(label2id)
logging_step = 100

# for lstm
num_train_epochs = 5  # epoch数
batch_size = 64  # mini-batch大小
max_seq_length = 128
learning_rate = 2e-5  # 学习率
dropout = 0.3  # 随机失活
patient = 10000  # 若超过1000batch效果还没提升，则提前结束训练
embedding_dim = 300  # 向量维度
hidden_size = 512  # lstm隐藏层
num_layers = 1  # lstm层数
bidirectional = True #
predict = True
max_vocab_size = 50000
wordvec_mode = "word2vec"
# eps = 1e-8

# for bert:
bert_path = curPath + '/pretrained_model'
if model_name == "bert":
    bert_path += '/bert-base-chinese'
elif model_name == "roberta":
    bert_path += '/chinese-roberta-wwm-ext'
cache_dir = ""
do_lower_case = True
gradient_accumulation_steps=1
max_grad_norm=1.0
warmup_proportion=0.1


