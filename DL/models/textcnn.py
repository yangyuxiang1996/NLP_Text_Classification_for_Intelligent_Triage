#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:       textcnn.py
@Time:       2021/08/27 17:34:19
@Author:     Yuxiang Yang
@Version:    1.0
@Describe:   
'''

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


