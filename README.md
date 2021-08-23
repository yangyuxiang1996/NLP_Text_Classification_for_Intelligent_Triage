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
