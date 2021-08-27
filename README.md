# 文本分类之智能医疗问诊

[TOC]

这个项目是利用不同科室患者的病情描述的数据集，采用机器学习或者深度学习模型实现文本分类。

## 框架

医疗问诊文本分类处理流程如下：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gtvqtp9kmcj61j60hu0tx02.jpg" alt="image-20210826160136042" style="zoom:50%;" />

## 数据清洗

原始数据集：数据源来于京东健康，任务是基于患者的病情描述，自动给一个门诊科室的分类

训练数据：28373 ；验证数据：1580 ；测试数据：1580

|   label    |  Num  |
| :--------: | :---: |
| 心血管内科 | 1195  |
|  小儿内科  | 2050  |
|    其它    | 2916  |
|    男科    | 10613 |
|    妇科    | 1305  |
|  耳鼻喉科  |  912  |
|  呼吸内科  | 2443  |
|  消化内科  | 1501  |
|  内分泌科  | 1144  |
|  神经内科  | 1191  |
|   皮肤科   | 3103  |

* 使用`re`去除标点符号等特殊字符：`pip install re`
* 中文分词：`pip install jieba`
* 长度大于1，有效减少特征维度
* 停用词过滤，过滤无用信息

清洗后数据示例：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gtvqubtntej614w09kwh002.jpg" alt="image-20210826171407825" style="zoom:50%;" />



## 特征工程

### 基于词向量的文本特征

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gtvquekmzdj60zo0q276h02.jpg" alt="image-20210826161308520" style="zoom:50%;" />

#### Word2Vec

* 获取词的embedding特征

* 考虑单词之间的顺序，使用不同的window_size（如2，3，4），对word embedding进行avg和max操作，并拼接在一起；

  ```python
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
  ```

#### TFIDF

* 获取word-text的tfidf权重

* `from sklearn.feature_extraction.text import TfidfVectorizer`

  ```python
  #count_vect 对 tfidfVectorizer 初始化
  logging.info("Training tfidf..........")
  count_vect = TfidfVectorizer(stop_words=self.stopwords,
                                  max_df=0.4,
                                  min_df=0.001,
                                  ngram_range=(1, 2))
  
  self.tfidf = count_vect.fit(self.train)
  self.train = [sample.split() for sample in self.train]
  ```

#### LDA

* 获取文本的主题特征

* `from gensim.models import LdaMulticore`

  ```python
  # 建立LDA模型
  logging.info("Training LDA model..........")
  self.LDAmodel = LdaMulticore(corpus=corpus,
                                  id2word=self.id2word,
                                  num_topics=30)
  ```

#### Label embedding

* 根据论文《Joint embedding of words and labels》获取标签空间的词嵌入

  ```python
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
  ```

#### Sentence embedding

```Python
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
```



### 基于人工定义的文本特征

* 词性： 考虑样本中词的词性，比如句子中各种词性(名词，动词的个数），使得构造的样本表示具有多样性 从而提高模型的分类精度。
* 标点符号
* 长度



### 特征拼接

将所有的特征拼接起来，用于模型训练

```python
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
```



## LightGBM

* 构建LightGBM模型：

```python
model_lgb = lgb.LGBMClassifier(boosting_type='gbdt',
                                objective='multiclass',
                                learning_rate=0.05,
                                num_leaves=32,
                                max_depth=5,
                                n_estimators=100,
                                n_jobs=1,
                                verbose=1)
```

* Grid Search调参

  ```python
  grid_cv_tuner = GridSearchCV(estimator=model,
                                  param_grid=params,
                                  scoring='accuracy',
                                  cv=5,
                                  verbose=3)
  grid_cv_tuner.fit(X_train, y_train)
  ```

* 结果：
  * accuracy: 0.829
  * prediction: 0.829
  * recall: 0.751
  * f1score: 0.775

## BiLSTM

机器学习需要人工提取大量的特征，这里使用BiLSTM模型作为深度学习模型分类的baseline，可以减少特征工程的步骤

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gtvquj539tj60qo0fuwfr02.jpg" alt="image-20210826172309707" style="zoom:50%;" />

```python
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes,
                 num_layers=2,
                 dropout=0.5,
                 bidirectional=True,
                 embeddings=None):
        super(LSTMModel, self).__init__()
        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size,
                                        embedding_dim,
                                        padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size,
                            batch_first=True,
                            num_layers=num_layers,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, x):
        embeddings = self.embedding(x[0])  # [batch_size, seq_len, embeding]=[128, 32, 300]
        hidden_state, _ = self.lstm(embeddings)
        hidden_state = self.dropout(hidden_state)
        output = torch.tanh(self.fc1(hidden_state[:, -1, :]))  # 句子最后时刻的 hidden state
        output = self.fc2(output)
        return output
```

* 结果：

  **batch_size为32，模型经过10个epoch训练后，准确率acc为82.53% 。**

## BERT

![image-20210827230417254](https://tva1.sinaimg.cn/large/008i3skNly1gtvqv4vlbqj61f00l20xq02.jpg)

直接使用预训练模型BERT进行微调，这里直接使用了Bert中的下游任务BertForSequenceClassification作为分类器，分类器的输出是token "[CLS]"位置的池化层输出，可以直接用于分类任务，首先构建模型：

```python
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
                       

class BertModelForMedical(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelForMedical, self).__init__(config)
        self.bert_model = BertForSequenceClassification(config)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,)
        pooled_output = outputs[0]
        return pooled_output

```

结果：

**batch_size为16，模型经过10个epoch训练后，准确率acc为84.05% 。**可以看出，还是比BiLSM强了点。

## RoBerta

尝试使用其它的Bert模型，如RoBerta，两次结果如下：

| epoch | batch_size |  Acc  |
| :---: | :--------: | :---: |
|   5   |     64     | 84.30 |
|   5   |     64     | 84.43 |

考虑到原始数据集数据量比较少，并且标签不均衡，可以采取的办法：

* 获取新数据，这里直接爬取了好大夫网的问诊数据，每个label各填充到了1w多条，保证label分布的均衡：

  |   label    |  Num  | Num (填充后) |
  | :--------: | :---: | :----------: |
  | 心血管内科 | 1195  |    10732     |
  |  小儿内科  | 2050  |    11405     |
  |    其它    | 2916  |   （删除）   |
  |    男科    | 10613 |    12964     |
  |    妇科    | 1305  |    11082     |
  |  耳鼻喉科  |  912  |    10427     |
  |  呼吸内科  | 2443  |    11771     |
  |  消化内科  | 1501  |    10953     |
  |  内分泌科  | 1144  |    10565     |
  |  神经内科  | 1191  |    10699     |
  |   皮肤科   | 3103  |     9913     |

  RoBerta在新数据集上训练的准确率：

  | epoch | batch_size |  Acc   |
  | :---: | :--------: | :----: |
  |   5   |     32     | 92.47% |

  果然数据才是王道！

## FastText

FastText是Facebook于2016年开源的一个词向量计算和文本分类工具，在学术上并没有太大创新。但是它的优点也非常明显，在文本分类任务中，fastText（浅层网络）往往能取得和深度网络相媲美的精度，却在训练时间上比深度网络快许多数量级。在标准的多核CPU上， 能够训练10亿词级别语料库的词向量在10分钟之内，能够分类有着30万多类别的50多万句子在1分钟之内。

1. 首先安装fasttext：`pip install fasttext`
2. 处理成fasttext要求的数据格式；
3. 训练fasttext

```python
def train_model(ipt=None, opt=None, dim=100, epoch=5, lr=0.1, loss='softmax', verbose=True):
    np.set_printoptions(suppress=True)
    start_time = time.time()
    classifier = fasttext.train_supervised(ipt,
                                           label='__label__',
                                           dim=dim,
                                           epoch=epoch,
                                           lr=lr,
                                           wordNgrams=2,
                                           loss=loss,
                                           verbose=verbose)
    """
        训练一个监督模型, 返回一个模型对象

        @param input:           训练数据文件路径
        @param lr:              学习率
        @param dim:             向量维度
        @param ws:              cbow模型时使用
        @param epoch:           次数
        @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
        @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
        @param minn:            构造subword时最小char个数
        @param maxn:            构造subword时最大char个数
        @param neg:             负采样
        @param wordNgrams:      n-gram个数
        @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
        @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
        @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
        @param lrUpdateRate:    学习率更新
        @param t:               负采样阈值
        @param label:           类别前缀
        @param verbose:         可视化
        @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
        @return model object
    """
    classifier.save_model(opt)
    logging.info('Model saved.')
    logging.info('used time: {:.4f}s'.format(time.time() - start_time))
    return classifier
```

* 训练结果：

  ```shell
  INFO - 17:35:18: Model saved.
  INFO - 17:35:18: used time: 4.2104s
  INFO - 17:35:18: Testing trained model.
  INFO - 17:35:18: precision: 0.9328, recall: 0.9328, f1: 0.9328
  INFO - 17:35:18: Testing trained model.
  INFO - 17:35:18: precision: 0.9244, recall: 0.9244, f1: 0.9244
  ```

* 训练速度<1min，f1 score直接到了0.9328。效果也是很惊人了。

* 跑几条测试看看：

  ```shell
  INFO - 17:47:36: text: 我 有 点 血压 高
  INFO - 17:47:36: clean text: 我 有 点 血压 高
  INFO - 17:47:36: used time: 0.0093s
  INFO - 17:47:36: label: 心血管内科, score: 0.9988077878952026
  
  INFO - 17:44:55: text: 我 头皮 好痒
  INFO - 17:44:55: clean text: 我 头皮 好痒
  INFO - 17:44:55: used time: 0.0005s
  INFO - 17:44:55: label: 皮肤科, score: 0.8614417314529419
  ```

* 文本分类任务比较简单的话，首选fasttext，训练速度快，准确率也很高。

## TextCNN

 [NLP\] 文本分类之TextCNN模型原理和实现(超详细)_GFDGFHSDS的博客-CSDN博客](https://blog.csdn.net/GFDGFHSDS/article/details/105295247)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gtvqxbwggbj60u00upq5f02.jpg" alt="img" style="zoom:67%;" />

code：

```python
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 feature_dim,
                 window_size, 
                 max_seq_length,
                 num_classes,
                 dropout=0.5,
                 embeddings=None,
                 fine_tune=True):
        super(TextCNN, self).__init__()
        assert isinstance(window_size, list) or isinstance(window_size, tuple)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embeddings is not None:
            logging.info("using preprained embeddings..........")
            logging.info("fine_tune embeddings: {}".format(fine_tune))
            self.embedding.from_pretrained(embeddings, freeze=fine_tune)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=feature_dim, 
                          kernel_size=h),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_seq_length-h+1)
            )
            for h in window_size
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=feature_dim*len(window_size),
                            out_features=num_classes)


    def forward(self, x):
        # x: input_ids
        input = x[0]
        embed_x = self.embedding(input)
        embed_x = embed_x.permute(0, 2, 1) # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        out = [conv(embed_x) for conv in self.convs]  #out[i]:batch_size x feature_size x 1
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = self.dropout(out)
        out = self.fc(out)

        return out
```



## 模型结果小结

| 模型     | 数据集 | 准确率             |
| -------- | ------ | ------------------ |
| LightGBM | origin | 82.9%              |
| BiLSTM   | origin | 82.53%             |
| Bert     | origin | 84.05%             |
| RoBerta  | origin | 84.43%             |
| RoBerta  | new    | 92.47%             |
| FastText | new    | 93.28%（f1 score） |
| TextCNN  | new    | 90.66%             |

## Trick

### 数据不均衡

* 上采样
* 下采样
* 数据增强

  * 回译
  * 同义词替换（word2vec，embedding）
  * tfidf替换
  * 文本生成
  * uda
  * focal loss

### 模型训练

* 数据shuffle
* optimizer
* schedular
* sampler
* batch_size
* lr
* padding
* gradient_accumulate_step

