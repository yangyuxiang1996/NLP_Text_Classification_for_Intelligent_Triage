#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 14:46:39
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-20 18:15:03
FilePath: /Chinese-Text-Classification/ml/embedding.py
Description: 
'''
import config
import gensim
from features import label2idx
from gensim.models import LdaMulticore
import jieba
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
import numpy as np
import pandas as pd
import logging
import os
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''

    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''
        # 停止词
        self.stopwords = []
        with open(config.stopwords, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                self.stopwords.append(line.strip())

        self.tfidf = None
        self.w2v = None
        self.LDAmodel = None

    def load_data(self, path):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        data = pd.read_csv(path, sep='\t', header=0)
        data = data.fillna("")

        # 对data['text']中的词进行分割，并去除停用词 参考格式： data['text'] = data["text"].apply(lambda x: " ".join(x))
        data['text'] = data['text'].apply(lambda x: " ".join(
            [w for w in x.split() if w not in self.stopwords and w != '']))

        self.labelToIndex = label2idx(data)
        data['label'] = data['label'].map(self.labelToIndex)
        data['label'] = data.apply(lambda row: float(row['label']), axis=1)
        data = data[['text', 'label']]

        # self.train, _, _ = np.split(data[['text', 'label']].sample(frac=1), [int(data.shape[0] * 0.7), int(data.shape[0] * 0.9)])
        self.train = data['text'].tolist()

        vocab = {}
        for sentence in self.train:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        with open(config.vocab_path, "w", encoding='utf-8') as f:
            for k, v in vocab.items():
                f.write("%s %s\n" % (k, v))

    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''
        #count_vect 对 tfidfVectorizer 初始化
        logging.info("Training tfidf..........")
        count_vect = TfidfVectorizer(stop_words=self.stopwords,
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))

        self.tfidf = count_vect.fit(self.train)
        self.train = [sample.split() for sample in self.train]

        #对 w2v 初始化 并建立词表，训练
        logging.info("Training word2vec..........")
        self.w2v = models.Word2Vec(sentences=self.train,
                                   min_count=2,
                                   window=5,
                                   vector_size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=4,
                                   max_vocab_size=50000)
        self.w2v.build_vocab(self.train, update=True)
        self.w2v.train(self.train,
                       total_examples=self.w2v.corpus_count,
                       epochs=15)

        self.id2word = gensim.corpora.Dictionary(self.train)
        corpus = [self.id2word.doc2bow(text) for text in self.train]
        logging.info(corpus[:5])

        # 建立LDA模型
        logging.info("Training LDA model..........")
        self.LDAmodel = LdaMulticore(corpus=corpus,
                                     id2word=self.id2word,
                                     num_topics=30)

    def saver(self):
        '''
        @description: save all model
        @param {type} None
        @return: None
        '''
        if not os.path.exists("model"):
            os.makedirs("model")

        joblib.dump(self.tfidf, './model/tfidf')

        self.w2v.wv.save_word2vec_format('./model/w2v.bin', binary=False)

        self.LDAmodel.save('./model/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        self.tfidf = joblib.load('./model/tfidf')
        self.w2v = models.KeyedVectors.load_word2vec_format(
            './model/w2v.bin', binary=False)
        self.lda = models.ldamodel.LdaModel.load('./model/lda')


if __name__ == "__main__":
    em = Embedding()
    em.load_data(config.train_data_file)
    em.trainer()
    em.saver()
