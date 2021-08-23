#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 14:46:39
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-20 14:46:47
FilePath: /Chinese-Text-Classification/ml copy/app.py
Description: 
'''

from flask import Flask, request
import json

from pandas.core.algorithms import mode
from model import Classifier


# 初始化模型， 避免在函数内部初始化，耗时过长
model = Classifier(mode='predict')
model.load()

# 初始化 flask
app = Flask(__name__)


#设定端口
@app.route('/predict', methods=["POST"])
def gen_ans():
    '''
    @description: 以RESTful的方式获取模型结果, 传入参数为title: 图书标题， desc: 图书描述
    @param {type}
    @return: json格式， 其中包含标签和对应概率
    '''
    result = {}
    result = {}
    text = request.form['text'] 

    label, score = model.predict(text)
    result = {
        "label": label
    }
    return json.dumps(result, ensure_ascii=False)


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
