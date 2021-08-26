# NLP_Text_Classification_for_Intelligent_Triage
这里记录了第一个NLP文本分类项目，目的是实现智能分诊


* lightGBM
{'learning_rate': 0.01, 'max_depth': 8} 0.8295562073854154 LGBMClassifier(learning_rate=0.01, max_depth=8, n_jobs=1, num_leaves=32, objective='multiclass', verbose=1)
learning_rate: 0.01
max_depth: 8
2021-08-20 22:07:10 model.py INFO saving model to ./model/clf_lgb......
2021-08-20 22:07:33 model.py INFO train accuracy: 0.8939132273640433
2021-08-20 22:07:34 model.py INFO prediction shape: (1580,)
2021-08-20 22:07:34 model.py INFO prediction example: 
 ['3' '10' '7' ... '1' '1' '6']
2021-08-20 22:07:34 model.py INFO prediction accuracy: 0.8291139240506329
2021-08-20 22:07:34 model.py INFO prediction precision: 0.8099336389601849
2021-08-20 22:07:34 model.py INFO prediction recall: 0.7518844413070671
2021-08-20 22:07:34 model.py INFO prediction f1score: 0.7759752185144976


* label
男科       12964
呼吸内科     11771
小儿内科     11405
妇科       11082
消化内科     10953
心血管内科    10732
神经内科     10699
内分泌科     10565
耳鼻喉科     10427
皮肤科       9913
其它        2621
Name: label, dtype: int64


男科       1650
呼吸内科     1444
小儿内科     1434
妇科       1431
心血管内科    1324
内分泌科     1322
消化内科     1318
皮肤科      1316
耳鼻喉科     1309
神经内科     1286
其它        307
Name: label, dtype: int64


男科       1519
呼吸内科     1502
妇科       1456
小儿内科     1440
消化内科     1398
内分泌科     1385
神经内科     1340
耳鼻喉科     1280
心血管内科    1274
皮肤科      1236
其它        311
Name: label, dtype: int64



## fasttext

INFO - 17:35:18: Model saved.
INFO - 17:35:18: used time: 4.2104s
INFO - 17:35:18: Testing trained model.
INFO - 17:35:18: precision: 0.9328, recall: 0.9328, f1: 0.9328
INFO - 17:35:18: Testing trained model.
INFO - 17:35:18: precision: 0.9244, recall: 0.9244, f1: 0.9244


INFO - 17:47:36: text: 我 有 点 血压 高
INFO - 17:47:36: clean text: 我 有 点 血压 高
INFO - 17:47:36: used time: 0.0093s
INFO - 17:47:36: label: 心血管内科, score: 0.9988077878952026

INFO - 17:44:55: text: 我 头皮 好痒
INFO - 17:44:55: clean text: 我 头皮 好痒
INFO - 17:44:55: used time: 0.0005s
INFO - 17:44:55: label: 皮肤科, score: 0.8614417314529419