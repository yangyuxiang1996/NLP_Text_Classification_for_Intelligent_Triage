#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 22:19:05
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-23 11:26:15
FilePath: /Chinese-Text-Classification/DL/utils/vocab.py
Description: 
'''
from collections import Counter
import config
import os
from gensim import models
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class Dictionary(object):
    def __init__(self, data, max_vocab_size=50000, min_count=None, start_end_tokens=False,
                 wordvec_mode=None, embedding_size=300):
        # 定义所需要参数
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.start_end_tokens = start_end_tokens
        self.embedding_size = embedding_size
        self.wordvec_mode = wordvec_mode
        self.PAD_TOKEN = '<PAD>'
        if isinstance(data, list):
            self.build_dictionary(data)
        else:
            data = data["text"].values.tolist()
            self.build_dictionary(data)

    def build_dictionary(self, data):
        # 构建词典主方法， 使用_build_dictionary构建
        self.vocab_words, self.word2idx, self.idx2word, self.idx2count = self._build_dictionary(data)
        self.vocabulary_size = len(self.vocab_words)

        if self.wordvec_mode is None:
            self.embedding = None
        elif self.wordvec_mode == 'word2vec':
            self.embedding = self._load_word2vec(data)
    
    def _load_word2vec(self, data):
        if os.path.exists(config.w2v_path):
            logging.info("加载 word2vec..........")
            self.w2v = models.KeyedVectors.load_word2vec_format(config.w2v_path, binary=True)   
        else:
            #对 w2v 初始化 并建立词表，训练
            logging.info("Training word2vec..........")
            data = [sentence.split() for sentence in data]
            logging.info("data: %s" % data[0])
            self.w2v = models.Word2Vec(sentences=data,
                                       min_count=2,
                                       window=5,
                                       size=config.embedding_dim,
                                       sample=6e-5,
                                       alpha=0.03,
                                       min_alpha=0.0007,
                                       negative=15,
                                       workers=4,
                                       max_vocab_size=config.max_vocab_size)
            self.w2v.build_vocab(data, update=True)
            self.w2v.train(data,
                           total_examples=self.w2v.corpus_count,
                           epochs=15)
            self.w2v.wv.save_word2vec_format(config.w2v_path, binary=True)

        word_vectors = np.random.random((self.vocabulary_size, config.embedding_dim))
        count = 0
        for word in self.vocab_words:
            index = self.word2idx[word]
            if word in self.w2v:
                count += 1
                word_vectors[index, :] = self.w2v.get_vector(word)
        print("count: %d" % count)
        return word_vectors        

    def indexer(self, word):
        # 根据词获取到对应的id
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def _build_dictionary(self, data):
        # 加入UNK标示， 按照需要加入EOS 或者EOS
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        vocab_size = 2 
        if self.start_end_tokens:
            vocab_words += ['<SOS>', '<EOS>']
            vocab_size += 2
        # 使用Counter 来同级次的个数
        counter = Counter(
            [word for sentence in data for word in sentence.split()])
        # 按照最大的词典个数进行筛选
        if self.max_vocab_size:
            counter = {word: freq for word, freq in counter.most_common(self.max_vocab_size - vocab_size)}
            print(len(counter))
        # 过滤掉低频词
        if self.min_count:
            counter = {word: freq for word, freq in counter.items() if freq >= self.min_count}

        # 按照出现次数进行排序， 并加到vocab_words 当中
        vocab_words += list(sorted(counter.keys()))

        idx2count = [counter.get(word, 0) for word in vocab_words]
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words
        return vocab_words, word2idx, idx2word, idx2count
