#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 19:03:12
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-23 11:40:59
FilePath: /Chinese-Text-Classification/DL/models/bilstm.py
Description: 
'''
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


