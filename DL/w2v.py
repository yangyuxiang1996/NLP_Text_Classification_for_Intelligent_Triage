#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:       w2v.py
@Time:       2021/08/27 17:36:35
@Author:     Yuxiang Yang
@Version:    1.0
@Describe:   
'''

import config
import re
from gensim import models
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def clean_txt(raw):
    fil = re.sub(r"[^a-zA-Z\u4e00-\u9fa5]+", ' ', raw).strip()
    return fil

def load_data(path, stopwords=None):
    '''
    @description:Load all data, then do word segment
    @param {type} None
    @return:None
    '''
    logging.info("read data..........")
    data = pd.read_csv(path, sep='\t', header=0)
    logging.info("data:\n %s" % data[:5])
    data = data.fillna("")

    data['text'] = data['text'].apply(lambda x: clean_txt(x))
    
    # 是否去除停用词
    if stopwords:
        data['text'] = data['text'].apply(lambda x: " ".join(
            [w for w in x.split() if w not in stopwords]))

    train = data['text'].values.tolist()
    
    vocab = {}
    for sentence in train:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    with open(config.vocab_path, "w", encoding='utf-8') as f:
        for k, v in vocab.items():
            f.write("%s\t%s\n" % (k, v))

    return train


def trainer(data):
    '''
    @description: Train tfidf,  word2vec, fasttext and autoencoder
    @param {type} None
    @return: None
    '''
    train = [sample.split() for sample in data]
    #对 w2v 初始化 并建立词表，训练
    logging.info("Training word2vec..........")
    w2v = models.Word2Vec(sentences=train,
                          min_count=2,
                          window=5,
                          size=300,
                          sample=6e-5,
                          alpha=0.03,
                          min_alpha=0.0007,
                          negative=15,
                          workers=4,
                          max_vocab_size=None)
    w2v.build_vocab(train, update=True)
    w2v.train(train,
              total_examples=w2v.corpus_count,
              epochs=15)
    return w2v

def saver(w2v):
    '''
    @description: save all model
    @param {type} None
    @return: None
    '''
    w2v.wv.save_word2vec_format('./data/w2v.bin', binary=False)

def load():
    '''
    @description: Load all embedding model
    @param {type} None
    @return: None
    '''
    w2v = models.KeyedVectors.load_word2vec_format(
        './data/w2v.bin', binary=False)
    return w2v


if __name__ == "__main__":
    stop_words = []
    with open(config.stopWords_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            stop_words.append(line)

    data = load_data(config.train_data_file, stopwords=stop_words)
    w2v = trainer(data)
    saver(w2v)


