#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 14:46:39
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-20 20:53:58
FilePath: /Chinese-Text-Classification/ml/features.py
Description: 
'''
import numpy as np
from numpy import linalg
import pandas as pd
import joblib
import string
import jieba.posseg as pseg
import jieba
import json
import os
import config
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def label2idx(data):
    # 加载所有类别， 获取类别的embedding， 并保存文件
    if os.path.exists(config.label2id_file):
        labelToIndex = json.load(open(config.label2id_file ,encoding='utf-8'))
    else:
        label = data['label'].unique()
        labelToIndex = dict(zip(label, list(range(len(label)))))
        with open(config.label2id_file, 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in labelToIndex.items()}, f)
    return labelToIndex


def get_tfidf(tfidf, data):
    stopWords = [x.strip() for x in open(config.stopwords, encoding='utf-8', mode='r').readlines()]
    text = data['text'].apply(lambda x: " ".join(
        [w for w in x.split() if w not in stopWords and w != '']))
    text = tfidf.transform(text.tolist()).toarray()
    # word = tfidf.get_feature_names()
    data_tfidf = pd.DataFrame(text)
    data_tfidf.columns = ['tfidf' + str(i) for i in range(data_tfidf.shape[1])]
    data = pd.concat([data, data_tfidf], axis=1)

    return data


def array2df(data, col):
    return pd.DataFrame.from_records(
        data[col].values,
        columns=[col + "_" + str(i) for i in range(len(data[col].iloc[0]))])


def get_embedding_feature(data, embedding_model):
    '''
    @description: , word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    @param {type}
    data, input data set
    @return:
    data, data set
    '''
    labelToIndex = label2idx(data)
    w2v_label_embedding = np.zeros(shape=(len(labelToIndex), embedding_model.vector_size))
    for i, key in enumerate(labelToIndex.keys()):
        vec = np.zeros(embedding_model.vector_size)
        for word in key:
            if word in embedding_model.key_to_index:
                vec += embedding_model[word]
        vec /= len(key)
        w2v_label_embedding[i] = vec
    joblib.dump(w2v_label_embedding, './data/w2v_label_embedding.pkl')
    # 根据未聚合的embedding 数据， 获取各类embedding 特征
    logging.info("transform w2v")
    tmp = data['text'].apply(lambda x: pd.Series(
        generate_feature(x, embedding_model, w2v_label_embedding)))
    tmp = pd.concat([array2df(tmp, col) for col in tmp.columns], axis=1)
    data = pd.concat([data, tmp], axis=1)
    return data


def wam(sentence, w2v_model, method='mean', aggregate=True):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    method： 聚合方法 mean 或者max
    aggregate: 是否进行聚合
    @return:
    '''
    # 获取句子中的词向量，放入list中
    stop_words = [x.strip() for x in open(config.stopwords, encoding='utf-8', mode='r').readlines()]
    arr = np.zeros((len(sentence.split()), 300))
    for i, word in enumerate(sentence.split()): 
        if word not in stop_words:
            try:
                v = w2v_model.get_vector(word)
                arr[i] = v
            except Exception:
                # print("word %s not found" % word)
                pass
            

    if not aggregate:
        return arr
    if len(arr) > 0:
        # 求平均
        if method == 'mean':
            return np.mean(arr, axis=0)
        # 最大值
        elif method == 'max':
            return np.max(arr, axis=0)
        else:
            raise NotImplementedError
    else:
        return np.zeros(300)


def rename_column(data, suffix):
    data.columns += suffix
    return data


def generate_feature(sentence, embedding_model, label_embedding):
    '''
    @description: word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    @param {type}
    data， input data, DataFrame
    label_embedding, all label embedding
    model_name, w2v means word2vec
    @return: data, DataFrame
    '''
    # 首先在预训练的词向量中获取标签的词向量句子,每一行表示一个标签表示
    # 每一行表示一个标签的embedding

    # 同上， 获取embedding 特征， 不进行聚合
    w2v = wam(sentence, embedding_model, aggregate=False)  # [seq_len * 300]

    if len(w2v) < 1:
        return {
            'w2v_label_mean': np.zeros(300),
            'w2v_label_max': np.zeros(300),
            'w2v_mean': np.zeros(300),
            'w2v_max': np.zeros(300),
            'w2v_2_mean': np.zeros(300),
            'w2v_3_mean': np.zeros(300),
            'w2v_4_mean': np.zeros(300),
            'w2v_2_max': np.zeros(300),
            'w2v_3_max': np.zeros(300),
            'w2v_4_max': np.zeros(300)
        }

    w2v_label_mean = Find_Label_embedding(w2v, label_embedding, method='mean')
    w2v_label_max = Find_Label_embedding(w2v, label_embedding, method='max')

    # 将embedding 进行max, mean聚合
    w2v_mean = np.mean(np.array(w2v), axis=0)

    w2v_max = np.max(np.array(w2v), axis=0)

    # 滑窗处理embedding 然后聚合
    w2v_2_mean = Find_embedding_with_windows(w2v, 2, method='mean')

    w2v_3_mean = Find_embedding_with_windows(w2v, 3, method='mean')

    w2v_4_mean = Find_embedding_with_windows(w2v, 4, method='mean')

    w2v_2_max = Find_embedding_with_windows(w2v, 2, method='max')

    w2v_3_max = Find_embedding_with_windows(w2v, 3, method='max')

    w2v_4_max = Find_embedding_with_windows(w2v, 4, method='max')

    return {
        'w2v_label_mean': w2v_label_mean,
        'w2v_label_max': w2v_label_max,
        'w2v_mean': w2v_mean,
        'w2v_max': w2v_max,
        'w2v_2_mean': w2v_2_mean,
        'w2v_3_mean': w2v_3_mean,
        'w2v_4_mean': w2v_4_mean,
        'w2v_2_max': w2v_2_max,
        'w2v_3_max': w2v_3_max,
        'w2v_4_max': w2v_4_max
    }


def softmax(x):
    '''
    @description: calculate softmax
    @param {type}
    x, ndarray of embedding
    @return: softmax result
    '''
    return np.exp(x) / np.exp(x).sum(axis=0)


def Find_Label_embedding(example_matrix, label_embedding, method='mean'):
    '''
    @description: 根据论文《Joint embedding of words and labels》获取标签空间的词嵌入
    @param {type}
    example_matrix(np.array 2D): denotes words embedding of input , (seq_len, embedding_size)
    label_embedding(np.array 2D): denotes the embedding of all label, (class_nums, embedding_size)
    @return: (np.array 1D) the embedding by join label and word
    '''
    # 根据矩阵乘法来计算label与word之间的相似度 cosin similiarity
    assert label_embedding.shape[1] == example_matrix.shape[1]
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
        np.linalg.norm(label_embedding) * np.linalg.norm(example_matrix)
    )
    # similarity_matrix = np.matmul(label_embedding, example_matrix.T) / (
    #     np.linalg.norm(label_embedding) * np.linalg.norm(example_matrix)) # (class_nums, seq_len)

    # 然后对相似矩阵进行均值池化，则得到了“类别-词语”的注意力机制
    # 这里可以使用max-pooling和mean-pooling,然后取softmax
    if method == 'mean':
        attention = np.mean(similarity_matrix, axis=1)  # (seq_len, 1)
    else:
        attention = np.max(similarity_matrix, axis=1)  # (seq_len, 1)
    attention = softmax(attention).reshape(-1, 1)

    # 将样本的词嵌入与注意力机制相乘得到
    attention_embedding = example_matrix * attention
    if method == 'mean':
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)


def Find_embedding_with_windows(embedding_matrix, window_size=2,
                                method='mean'):
    '''
    @description: generate embedding use window
    @param {type}
    embedding_matrix, input sentence's embedding
    window_size, 2, 3, 4
    method, max/ mean
    @return: ndarray of embedding
    '''
    # 最终的词向量
    result_list = []
    # 遍历input的长度， 根据窗口的大小获取embedding， 进行mean操作， 然后将得到的结果extend到list中， 最后进行mean max 聚合
    for k1 in range(len(embedding_matrix)):
        # 如何当前位置 + 窗口大小 超过input的长度， 则取当前位置到结尾
        # mean 操作后要reshape 为 （1， 300）大小
        if int(k1 + window_size) > len(embedding_matrix):
            result_list.append(np.mean(embedding_matrix[k1:len(embedding_matrix)], axis=0))
        else:
            result_list.append(np.mean(embedding_matrix[k1:k1 + window_size], axis=0))

    result_list = np.array(result_list).reshape(-1, 300)

    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def get_lda_features_helper(lda_model, document):
    '''
    @description: Transforms a bag of words document to features.
    It returns the proportion of how much each topic was
    present in the document.
    @param {type}
    lda_model: lda_model
    document, input
    @return: lda feature
    '''
    # 基于bag of word 格式数据获取lda的特征
    topic_importance = lda_model.get_document_topics(document, minimum_probability=0)

    topic_importance = np.array(topic_importance)
    return topic_importance[:, 1]


def get_lda_features(data, LDAmodel):
    if isinstance(data.iloc[0]['text'], str):
        data['text'] = data['text'].apply(lambda x: x.split())
    data['bow'] = data['text'].apply(
        lambda x: LDAmodel.id2word.doc2bow(x))
    # logging.info("data['bow']:\n %s", data['bow'].head(5))
    data['lda'] = list(
        map(lambda doc: get_lda_features_helper(LDAmodel, doc), data['bow']))
    # logging.info("data['lda']:\n %s", data['lda'].head(5))
    cols = [x for x in data.columns if x not in ['lda', 'bow']] 
    return pd.concat([data[cols], array2df(data, 'lda')], axis=1)


def tag_part_of_speech(data):
    '''
    @description: tag part of speech, then calculate the num of noun, adj and verb
    @param {type}
    data, input data
    @return:
    noun_count,num of noun
    adjective_count, num of adj
    verb_count, num of verb
    '''
    # 获取文本的词性， 并计算名词，动词， 形容词的个数
    words = [tuple(x) for x in list(pseg.cut(data))]
    noun_count = len(
        [w for w in words if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    adjective_count = len([w for w in words if w[1] in ('JJ', 'JJR', 'JJS')])
    verb_count = len([
        w for w in words if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    ])
    return noun_count, adjective_count, verb_count


ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def get_basic_feature_helper(text):
    '''
    @description: get_basic_feature, length, capitals number, num_exclamation_marks, num_punctuation, num_question_marks, num_words, num_unique_words .etc
    @param {type}
    df, dataframe
    @return:
    df, dataframe
    '''
    if isinstance(text, str):
        text = text.split()
    # 分词
    queryCut = [i if i not in ch2en.keys() else ch2en[i] for i in text]
    # 词的个数
    num_words = len(queryCut)

    # 大写的个数
    capitals = sum(1 for c in queryCut if c.isupper())
    # 大写 与 文词的个数的占比
    caps_vs_length = capitals / num_words
    # 感叹号的个数
    num_exclamation_marks = queryCut.count('!')
    # 问号个数
    num_question_marks = queryCut.count('?')

    # 标点符号个数
    num_punctuation = sum(queryCut.count(w) for w in string.punctuation)
    # *&$%字符的个数
    num_symbols = sum(queryCut.count(w) for w in ['*', '&', '$', '%'])


    # 唯一词的个数
    num_unique_words = len(set(w for w in queryCut))
    # 唯一词 与总词数的比例
    words_vs_unique = num_unique_words / num_words
    # 获取名词， 形容词， 动词的个数， 使用tag_part_of_speech函数
    nouns, adjectives, verbs = tag_part_of_speech("".join(text))
    # 名词占词的个数的比率
    nouns_vs_length = nouns / num_words
    # 形容词占词的个数的比率
    adjectives_vs_length = adjectives / num_words
    # 动词占词的个数的比率
    verbs_vs_length = verbs / num_words
    # 首字母大写其他小写的个数
    count_words_title = len([w for w in queryCut if w.istitle()])
    # 平均词的个数
    mean_word_len = np.mean([len(w) for w in queryCut])
    return {
        'num_words': num_words,
        'capitals': capitals,
        'caps_vs_length': caps_vs_length,
        'num_exclamation_marks': num_exclamation_marks,
        'num_question_marks': num_question_marks,
        'num_punctuation': num_punctuation,
        'num_symbols': num_symbols,
        'num_unique_words': num_unique_words,
        'words_vs_unique': words_vs_unique,
        'nouns': nouns,
        'adjectives': adjectives,
        'verbs': verbs,
        'nouns_vs_length': nouns_vs_length,
        'adjectives_vs_length': adjectives_vs_length,
        'verbs_vs_length': verbs_vs_length,
        'count_words_title': count_words_title,
        'mean_word_len': mean_word_len
    }


def get_basic_feature(data):
    tmp = data['text'].apply(
        lambda x: pd.Series(get_basic_feature_helper(x)))
    return pd.concat([data, tmp], axis=1)
