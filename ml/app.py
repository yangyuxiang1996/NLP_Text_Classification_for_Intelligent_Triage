'''
Author: xiaoyao jiang
LastEditors: Peixin Lin
Date: 2020-08-31 14:18:26
LastEditTime: 2021-01-03 21:41:09
FilePath: /JD_NLP1-text_classfication/app.py
Desciption: Application.
'''
from flask import Flask, request
import json

from pandas.core.algorithms import mode
from model import Classifier


#######################################################################
#          TODO:  Initialize and load classifier model      #
#######################################################################
# 初始化模型， 避免在函数内部初始化，耗时过长
model = Classifier(mode='predict')
model.load()

#######################################################################
#          TODO:  Initialize flask     #
#######################################################################
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
    #######################################################################
    #          TODO:  预测结果并返回 #
    #######################################################################
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
