#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 19:03:23
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-23 22:08:47
FilePath: /Chinese-Text-Classification/DL/models/bert_model.py
Description: 
'''
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
                       

class BertModelForMedical(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelForMedical, self).__init__(config)
        self.bert_model = BertForSequenceClassification(config)
#         for param in self.bert_model.parameters():
#             param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,)
        pooled_output = outputs[0]
#         out = self.fc(self.dropout(pooled_output))
        return pooled_output

