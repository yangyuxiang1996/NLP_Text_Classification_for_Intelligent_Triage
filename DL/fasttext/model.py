#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-25 17:04:45
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-25 20:12:07
FilePath: /Chinese-Text-Classification/DL/fasttext/model.py
Description: 
'''
import fasttext
import numpy as np
import os
import json
import time
import sys
import logging
import jieba
from data import clean_txt
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def train_model(ipt=None, opt=None, dim=100, epoch=5, lr=0.1, loss='softmax', verbose=True):
    np.set_printoptions(suppress=True)
    start_time = time.time()
    classifier = fasttext.train_supervised(ipt,
                                           label='__label__',
                                           dim=dim,
                                           epoch=epoch,
                                           lr=lr,
                                           wordNgrams=2,
                                           loss=loss,
                                           verbose=verbose)
    """
        训练一个监督模型, 返回一个模型对象

        @param input:           训练数据文件路径
        @param lr:              学习率
        @param dim:             向量维度
        @param ws:              cbow模型时使用
        @param epoch:           次数
        @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
        @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
        @param minn:            构造subword时最小char个数
        @param maxn:            构造subword时最大char个数
        @param neg:             负采样
        @param wordNgrams:      n-gram个数
        @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
        @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
        @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
        @param lrUpdateRate:    学习率更新
        @param t:               负采样阈值
        @param label:           类别前缀
        @param verbose:         可视化
        @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
        @return model object
    """
    classifier.save_model(opt)
    logging.info('Model saved.')
    logging.info('used time: {:.4f}s'.format(time.time() - start_time))
    return classifier


def test(classifier, model_test_file):
    '''
    @description: 验证模型
    @param {type}
    classifier： model
    model_test_file： 测试数据路径
    @return:
    '''
    logging.info('Testing trained model.')
    result = classifier.test(model_test_file)  # (测试数据量，precision，recall)

    # F1 score
    f1 = result[1] * result[2] * 2 / (result[2] + result[1])
    logging.info("precision: %.4f, recall: %.4f, f1: %.4f" % (result[1], result[2], f1))


def predict(text, model_path):
    '''
    @description: 预测
    @param {type}
    text： 文本
    @return: label, score
    '''
    logging.info('Predicting.')
    classifier = fasttext.load_model(model_path)
    clean_text = clean_txt(text)
    logging.info('text: %s' % text)
    logging.info('clean text: %s' % clean_text)
    start_time = time.time()
    label, score = classifier.predict(clean_text)
    logging.info('used time: {:.4f}s'.format(time.time() - start_time))
    return label, score


if __name__ == "__main__":
    train_path = "../data/train_fast.txt"
    eval_path = "../data/eval_fast.txt"
    test_path = "../data/test_fast.txt"
    label2id = json.load(open("../data/label2id.json", 'r'))
    id2label = {v:k for k, v in label2id.items()}
    dim = 100
    lr = 0.1
    epoch = 5
    model_path = f'fasttext_dim{str(dim)}_lr{str(lr)}_iter{str(epoch)}.bin'
    train = int(sys.argv[1])
    if train:
        model = train_model(ipt=train_path,
                            opt=model_path,
                            dim=dim,
                            epoch=epoch,
                            lr=lr,
                            loss='softmax')

        test(model, eval_path)
        test(model, test_path)
    else:
        stop_words = []
        with open("../data/stopwords.txt", 'r') as f:
            for line in f.readlines():
                line = line.strip()
                stop_words.append(line)
        text = "您好 医生  我 一 热 或者 紧张 着急 的 时候  身上 各处 就 会痒  而且 不是 持续 那 种 痒  是 针扎 一下 的 那种  这 地方 痒 一下 那 地方 痒 一下 的".replace(" ", "")
        text = " ".join([word for word in jieba.cut(text) if word not in stop_words])
        text = clean_txt(text)
        label, score = predict(text, model_path)
        label = id2label[label[0].replace("__label__", "")]
        logging.info("label: {}, score: {}".format(label, score[0]))
    

