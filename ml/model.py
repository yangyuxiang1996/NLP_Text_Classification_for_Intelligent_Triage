#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 14:46:39
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-26 17:24:58
FilePath: /Chinese-Text-Classification/ml/model.py
Description: 
'''

import json
from logging import log
import joblib
import os
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
from tools import bayes_cv_search, grid_cv_search, create_logger, bayes_optimize
from skopt.space import Real, Categorical, Integer
from sklearn.decomposition import PCA 
import config
import pickle
from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,
                      get_lda_features, get_tfidf)
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class Classifier:
    def __init__(self, mode=None, tuner=None, model_name="lgb") -> None:
        self.stopWords = [
            x.strip() for x in open(config.stopwords, encoding='utf-8', mode='r').readlines()
        ]
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(
            open(config.label2id_file, encoding='utf-8'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}
        self.mode = mode
        if not self.mode:
            self.mode = 'train'
        self.tuner = tuner
        assert self.mode in ['train', 'predict']
        if self.tuner:
            assert self.tuner in ['bayes', 'grid']
        if self.mode == "train":
            self.train_data = pd.read_csv(config.train_data_file, sep='\t').dropna().reset_index(drop=True)
            self.dev_data = pd.read_csv(config.eval_data_file, sep='\t').dropna().reset_index(drop=True)
        else:
            self.test_data = pd.read_csv(config.test_data_file, sep='\t').dropna().reset_index(drop=True)
        self.exclusive_col = ['text', 'lda', 'bow', 'label']
        self.model=None
        self.model_name = model_name

    def feature_engineer(self, data):
        logging.info("start feature engineering......")
        logging.info("data:\n %s" % data)
        data['label'] = data['label'].apply(lambda x: self.labelToIndex[x])
        data = get_tfidf(self.embedding.tfidf, data)
        data = get_embedding_feature(data, self.embedding.w2v)
        data = get_lda_features(data, self.embedding.lda)
        data = get_basic_feature(data)
        logging.info("data:\n %s" %  data)
        return data

    def train(self):
        if os.path.exists(config.cached_train_data):
            self.train_data = pickle.load(open(config.cached_train_data, "rb"))
        else:
            self.train_data = self.feature_engineer(self.train_data)
            pickle.dump(self.train_data, open(config.cached_train_data, "wb"))
        
        if os.path.exists(config.cached_eval_data):
            self.dev_data = pickle.load(open(config.cached_eval_data, "rb"))
        else:
            self.dev_data = self.feature_engineer(self.dev_data)
            pickle.dump(self.dev_data, open(config.cached_eval_data, "wb"))

        logging.info("train_data: \n", self.train_data[:5])
        logging.info("eval_data: \n", self.dev_data[:5])
        cols = [x for x in self.train_data.columns if x not in self.exclusive_col]

        X_train = self.train_data[cols]
        y_train = self.train_data['label']

        X_test = self.dev_data[cols]
        y_test = self.dev_data['label']

        # #初始化多标签训练
        logging.info("start training......")
        # mlb = MultiLabelBinarizer(sparse_output=False)

        # y_train = mlb.fit_transform([[label] for label in y_train.values])
        # logging.info('class: %s', mlb.classes_)
        # y_test = mlb.transform([[label] for label in y_test.values])

        logging.info('X_train shape: %s', X_train.shape)
        logging.info('y_train class: %s', y_train.shape)
        logging.info('y_train: %s', y_train.value_counts())

        # 初始化训练参数，并进行fit
        # todo：选择不同机器学习模型
        if self.mode == "train":
            model_lgb = lgb.LGBMClassifier(boosting_type='gbdt',
                                           objective='multiclass',
                                           learning_rate=0.05,
                                           num_leaves=32,
                                           max_depth=5,
                                           n_estimators=100,
                                           n_jobs=1,
                                           verbose=1)

            model_svc = SVC(random_state=2021,verbose=1)
            model_lr = LogisticRegression(random_state=2021)
            model_rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=2021)
            # 贝叶斯搜索
            if self.tuner and self.tuner == 'bayes':
                logging.info("use bayesian optimization")
                train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
                best_params = bayes_optimize(train_data, opt_round=15)
                best_params['params']['num_leaves'] = int(best_params['params']['num_leaves'])
                best_params['params']['max_depth'] = int(best_params['params']['max_depth'])
                logging.info("best param", best_params)
                self.model = model_lgb.set_params(**best_params['params'])
                logging.info('fit model ')
                self.model.fit(X_train, y_train)

            elif self.tuner and self.tuner == 'grid':
                params_grid = {
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [8, 10]
                }
                clf = grid_cv_search(model=model_lgb, params=params_grid, X_train=X_train, y_train=y_train)
                self.model = clf.best_estimator_
            else:
                if self.model_name == 'lgb':
                    self.model = model_lgb
                elif self.model_name == 'svc':
                    self.model = model_svc
                elif self.model_name == 'rf':
                    self.model = model_rf
                elif self.model_name == 'lr':
                    self.model = model_lr
                self.model.fit(X_train, y_train)
            
            self.save()

        else:
            self.load()

        y_pred = self.model.predict(X_train)
        logging.info('train accuracy: %s', metrics.accuracy_score(y_train, y_pred))
        prediction = self.model.predict(X_test)
        logging.info('prediction shape: %s', prediction.shape)
        logging.info('prediction example: \n %s', prediction)
        logging.info('prediction accuracy: %s', metrics.accuracy_score(y_test, prediction))
        logging.info('prediction precision: %s', metrics.precision_score(y_test, prediction, average="macro"))
        logging.info('prediction recall: %s', metrics.recall_score(y_test, prediction, average="macro"))
        logging.info('prediction f1score: %s', metrics.f1_score(y_test, prediction, average="macro"))


    def save(self):
        logging.info('saving model to ./model/clf_{}......'.format(self.model_name))
        joblib.dump(self.model, './model/clf_{}'.format(self.model_name))

    def load(self):
        logging.info('loading model from ./model/clf_{}......'.format(self.model_name))
        self.model = joblib.load('./model/clf_{}'.format(self.model_name))

    def predict(self, text, desc=None):
        if isinstance(text, str):
            text = [text]
        df = pd.DataFrame(text, columns=['text'])
        df['text'] = df['text'].apply(lambda x: " ".join(
            [w for w in x.split(' ') if w not in self.stopWords and w != '']))
        df = get_tfidf(self.embedding.tfidf, df)
        df = get_embedding_feature(df, self.embedding.w2v)
        df = get_lda_features(df, self.embedding.lda)
        df = get_basic_feature(df)
        cols = [x for x in df.columns if x not in self.exclusive_col]

        # 利用模型获得预测结果
        if self.model is None:
            self.load()
        pred = self.model.predict(df[cols])
        return [self.ix2label[pred[i]] for i in range(len(pred))]


if __name__ == "__main__":
    model = Classifier(mode='train', tuner="grid", model_name="lgb")
    model.train()
    
    bc = Classifier(mode='predict')
    df = pd.read_csv("./data/test.csv", sep='\t').dropna().reset_index(drop=True)
    text = df['text'].values
    target = df['label']
    label = bc.predict(text)

    print("predicted label: %s " % label)
    logging.info('prediction accuracy: %s', metrics.accuracy_score(target, label))