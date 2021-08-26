#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 14:46:39
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-26 17:10:52
FilePath: /Chinese-Text-Classification/ml/data.py
Description: 数据预处理
'''
import config
import json

id2label = {}
with open(config.id2label_file, 'r') as txt:
    id2label = json.load(txt)

stopwords = []
with open(config.stopwords, 'r') as txt:
    for line in txt.readlines():
        line = line.strip()
        stopwords.append(line)

print(id2label)
for filepath in [config.train_raw_file,config.eval_raw_file,config.test_raw_file]:
    samples = []
    with open(filepath, 'r') as txt:
        for line in txt.readlines():
            ID, text = line.strip().split('\t')
            label = id2label[ID]
            text = [word for word in text.split(' ') if word != '||' and word not in stopwords and len(word) > 1]
            sample = label+'\t'+" ".join(text)
            samples.append(sample)

    outfile = config.train_data_file
    if 'eval' in filepath:
        outfile = config.eval_data_file
    if 'test' in filepath:
        outfile = config.test_data_file

    with open(outfile, 'w') as csv:
        csv.write('label\ttext\n')
        for sample in samples:
            csv.write(sample)
            csv.write('\n')
