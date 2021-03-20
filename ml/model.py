'''
Author: xiaoyao jiang
LastEditors: Peixin Lin
Date: 2020-08-31 14:19:30
LastEditTime: 2021-01-03 21:36:09
FilePath: /JD_NLP1-text_classfication/model.py
Desciption:
'''
import json
import jieba
import joblib
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

from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,
                      get_lda_features, get_tfidf)

# import logging
# logging.basicConfig(
#          filename='train.log',
#          filemode='w',
#          level=logging.INFO,
#          format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
#          datefmt='%H:%M:%S',
#  )
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# # tell the handler to use this format
# console.setFormatter(formatter)
# # add the handler to the root logger
# logging.getLogger('').addHandler(console)
logger = create_logger('train.log')


class Classifier:
    def __init__(self, mode=None, tuner=None, model_name="lgb") -> None:
        self.stopWords = [
            x.strip() for x in open('./data/stopwords.txt', encoding='utf-8', mode='r').readlines()
        ]
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(
            open('./data/label2id.json', encoding='utf-8'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}
        self.mode = mode
        if not self.mode:
            self.mode = 'train'
        self.tuner = tuner
        assert self.mode in ['train', 'eval', 'predict']
        if self.tuner:
            assert self.tuner in ['bayes', 'grid']
        if self.mode == "train" or self.mode == "eval":
            self.train = pd.read_csv('./data/train.csv', sep='\t').dropna().reset_index(drop=True)
            self.dev = pd.read_csv('./data/eval.csv', sep='\t').dropna().reset_index(drop=True)
            self.test = pd.read_csv('./data/test.csv', sep='\t').dropna().reset_index(drop=True)
        self.exclusive_col = ['text', 'lda', 'bow', 'label']
        self.model=None
        self.model_name = model_name

    def feature_engineer(self, data):
        logger.info("start feature engineering......")
        data['label'] = data['label'].apply(lambda x: self.labelToIndex[x])
        # logging.info(data)
        data = get_tfidf(self.embedding.tfidf, data)
        # logging.info(data)
        data = get_embedding_feature(data, self.embedding.w2v)
        # logging.info(data)
        data = get_lda_features(data, self.embedding.lda)
        # logging.info(data)
        data = get_basic_feature(data)
        logger.info("data:\n %s", data)
        return data

    def trainer(self):
        self.train = self.feature_engineer(self.train)
        self.dev = self.feature_engineer(self.dev)
        cols = [x for x in self.train.columns if x not in self.exclusive_col]

        X_train = self.train[cols]
        # y_train = train['label'].apply(lambda x: eval(x))
        y_train = self.train['label']
        # print(y_train.tolist())

        X_test = self.dev[cols]
        # y_test = dev['label'].apply(lambda x: eval(x))
        y_test = self.dev['label']
        # print(y_test)

        #######################################################################
        #          TODO:  lgb模型训练 #
        #######################################################################
        # #初始化多标签训练
        logger.info("start training......")
        #         mlb = MultiLabelBinarizer(sparse_output=False)

        #         y_train = mlb.fit_transform([[label] for label in y_train.values])
        #         logging.info('class: %s', mlb.classes_)
        #         y_test = mlb.transform([[label] for label in y_test.values])

        logger.info('X_train shape: %s', X_train.shape)
        logger.info('y_train class: %s', y_train.shape)
        logger.info('y_train: %s', y_train.value_counts())

        #######################################################################
        #          TODO:  lgb模型训练 #
        #######################################################################
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
                # search_spaces = {
                #     'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                #     'num_leaves': Integer(2, 500),
                #     'max_depth': Integer(0, 500),
                #     'min_child_samples': Integer(0, 200),
                #     'max_bin': Integer(100, 100000),
                #     'subsample': Real(0.01, 1.0, 'uniform'),
                #     'subsample_freq': Integer(0, 10),
                #     'colsample_bytree': Real(0.01, 1.0, 'uniform'),
                #     'min_child_weight': Real(0, 10),
                #     'subsample_for_bin': Integer(100000, 500000),
                #     'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
                #     'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
                #     'scale_pos_weight': Real(1e-6, 500, 'log-uniform'),
                #     'n_estimators': Integer(10, 10000),
                # }
                # logger.info('bayes search space: %s' % search_spaces)


                # clf = bayes_cv_search(model_lgb, search_spaces, X_train, y_train)

                # # self.model.fit(X_train, y_train)
                # self.model = clf.best_estimator_

                logger.info("use bayesian optimization")
                train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
                best_params = bayes_optimize(train_data, opt_round=15)
                best_params['params']['num_leaves'] = int(best_params['params']['num_leaves'])
                best_params['params']['max_depth'] = int(best_params['params']['max_depth'])
                logger.info("best param", best_params)
                self.model = model_lgb.set_params(**best_params['params'])
                logger.info('fit model ')
                self.model.fit(X_train, y_train)

            elif self.tuner and self.tuner == 'grid':
                params_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [5, 6, 7, 8, 9, 10]
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
        logger.info('train accuracy: %s', metrics.accuracy_score(y_train, y_pred))
        prediction = self.model.predict(X_test)
        logger.info('prediction shape: %s', prediction.shape)
        logger.info('prediction example: \n %s', prediction)
        logger.info('prediction accuracy: %s', metrics.accuracy_score(y_test, prediction))
        logger.info('prediction precision: %s', metrics.precision_score(y_test, prediction, average="macro"))
        logger.info('prediction recall: %s', metrics.recall_score(y_test, prediction, average="macro"))
        logger.info('prediction f1score: %s', metrics.f1_score(y_test, prediction, average="macro"))


    def save(self):
        logger.info('saving model to ./model/clf_{}......'.format(self.model_name))
        joblib.dump(self.model, './model/clf_{}'.format(self.model_name))

    def load(self):
        logger.info('loading model from ./model/clf_{}......'.format(self.model_name))
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
        #######################################################################
        #          TODO:  lgb模型预测 #
        #######################################################################
        # 利用模型获得预测结果
        if self.model is None:
            self.load()
        pred = self.model.predict(df[cols])
        return [self.ix2label[pred[i]] for i in range(len(pred))]


if __name__ == "__main__":
    bc = Classifier(mode='train', tuner=None, model_name="svc")
    bc.trainer()

    # bc = Classifier(mode='predict')
    # df = pd.read_csv("./data/test.csv", sep='\t').dropna().reset_index(drop=True)
    # text = df['text'].values
    # target = df['label']
    # label = bc.predict(text)


    # print("predicted label: %s " % label)
    # logger.info('prediction accuracy: %s', metrics.accuracy_score(target, label))