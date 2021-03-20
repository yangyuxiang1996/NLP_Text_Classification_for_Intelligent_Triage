from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import lightgbm as lgb
import logging
import json
import re
import time
from datetime import timedelta
from logging import handlers
from bayes_opt import BayesianOptimization


def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger


def grid_cv_search(model, params, X_train, y_train):

    grid_cv_tuner = GridSearchCV(estimator=model,
                                 param_grid=params,
                                 scoring='accuracy',
                                 cv=5,
                                 verbose=3)
    grid_cv_tuner.fit(X_train, y_train)
    print(grid_cv_tuner.best_params_, grid_cv_tuner.best_score_, grid_cv_tuner.best_estimator_)

    for param_name in sorted(grid_cv_tuner.best_params_.keys()):
        print("%s: %r" % (param_name, grid_cv_tuner.best_params_[param_name]))

    return grid_cv_tuner


def bayes_cv_search(model, params, X_train, y_train):

    bayes_cv_tuner = BayesSearchCV(estimator=model,
                                   search_spaces=params,
                                   scoring='f1_macro',
                                   cv=StratifiedKFold(n_splits=5),
                                   n_iter=30,
                                   verbose=1,
                                   refit=True)

    bayes_cv_tuner.fit(X_train, y_train)

    print(bayes_cv_tuner.best_params_, bayes_cv_tuner.best_score_, bayes_cv_tuner.best_estimator_)

    for param_name in sorted(bayes_cv_tuner.best_params_.keys()):
        print("%s: %r" % (param_name, bayes_cv_tuner.best_params_[param_name]))

    return bayes_cv_tuner


def bayes_optimize(train_data, init_round=3, opt_round=5, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05):
    
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        
        params = {'application': 'multiclass',
                  'num_iterations': n_estimators,
                  'learning_rate': learning_rate,
                  'early_stopping_round': 100,
                  'num_class': len(json.load(open('../../dataset/data/label2id.json', encoding='utf-8'))),
                  'metric': 'multi_logloss',
                  'num_leaves': int(round(num_leaves)),
                  'feature_fraction': max(min(feature_fraction, 1), 0),
                  'bagging_fraction': max(min(bagging_fraction, 1), 0),
                  'max_depth': int(round(max_depth)),
                  'lambda_l1':  max(lambda_l1, 0),
                  'lambda_l2': max(lambda_l2, 0),
                  'min_split_gain': min_split_gain,
                  'min_child_weight': min_child_weight
                  }
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval=200)
        
        return max(cv_result['multi_logloss-mea'])
    
    optimizer = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45), 
                                                'feature_fraction': (0.1, 0.9), 
                                                'bagging_fraction': (0.8, 1),
                                                'max_depth': (5, 8.99), 
                                                'lambda_l1': (0, 5),
                                                'lambda_l2': (0, 3),
                                                'min_split_gain': (0.001, 0.1),
                                                'min_child_weight': (5, 50)},
                                     random_state=0)
    # optimize
    optimizer.maximize(init_points=init_round, n_iter=opt_round)
    # return best parameters
    return optimizer.max
