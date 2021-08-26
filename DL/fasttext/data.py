#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-25 16:37:55
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-25 17:48:21
FilePath: /Chinese-Text-Classification/DL/fasttext/data.py
Description: 
'''
import pandas as pd
import json
import re
import os

def clean_txt(raw):
    fil = re.sub(r"[^a-zA-Z\u4e00-\u9fa5]+", ' ', raw).strip()
    return fil

def prepare_data(data_path, label_path):
    data = pd.read_csv(data_path, sep='\t', header=0)
    print(data[:5])
    label2id = json.load(open(label_path, 'r'))
    id2label = {v:k for k, v in label2id.items()}
    print(label2id, id2label)
    data['text'] = data['text'].apply(lambda x: clean_txt(x))
    if "train" in data_path:
        output = os.path.join(os.path.dirname(data_path), 'train_fast.txt')
    elif "test" in data_path:
        output = os.path.join(os.path.dirname(data_path), 'test_fast.txt')
    else:
        output = os.path.join(os.path.dirname(data_path), 'eval_fast.txt')
    labels = data['label'].values.tolist()
    texts = data['text'].values.tolist()
    count = 0
    with open(output, mode='w', encoding='utf-8') as f:
        for label, text in zip(labels, texts):
            line = '__label__{}\t{}'.format(label2id[label], text)
            if count < 5:
                print(line)
            f.write(line+'\n')
            count += 1

if __name__ == '__main__':
    # prepare_data("../data/train_new.csv", "../data/label2id_new.json")
    # prepare_data("../data/eval_new.csv", "../data/label2id_new.json")
    # prepare_data("../data/test_new.csv", "../data/label2id_new.json")
    prepare_data("../data/train_new.csv", "../data/label2id.json")
    prepare_data("../data/eval.csv", "../data/label2id.json")
    prepare_data("../data/test.csv", "../data/label2id.json")



