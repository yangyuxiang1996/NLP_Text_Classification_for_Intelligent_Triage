#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-20 19:02:58
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-23 16:24:20
FilePath: /Chinese-Text-Classification/DL/utils/dataset.py
Description: 
'''
import sys
sys.path.append("../")
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
import config
from transformers import RobertaTokenizer, BertTokenizer, BertModel
from utils.vocab import Dictionary
from transformers import AutoTokenizer, AutoModelForMaskedLM


class MedicalData(Dataset):
    def __init__(self,
                 path,
                 max_length=128,
                 tokenizer=None,
                 char_level=True,
                 dictionary=None):
        self.data = pd.read_csv(path, sep='\t', header=0)
        if char_level:
            self.data["text"] = self.data["text"].apply(
                lambda x: "".join(x.split(" ")))
        print(self.data['text'])
        self.label2id = json.load(open(config.label2id_file, "r"))
        self.id2label = json.load(open(config.id2label_file, "r"))
        self.data["label"] = self.data["label"].apply(
            lambda x: self.label2id[x])
        print("Label: ")
        print(self.data['label'].value_counts())
        self.embeddings = None
        self.tokenizer = tokenizer
        if "bert" not in config.model_name:
            if dictionary is None:
                self.tokenizer = Dictionary(self.data,
                                            start_end_tokens=True,
                                            wordvec_mode=config.wordvec_mode)
            else:
                self.tokenizer = dictionary
            self.embeddings = self.tokenizer.embedding
        
        self.max_length = max_length

    def __getitem__(self, i):
        text, label = self.data["text"].iloc[i], int(
            self.data["label"].iloc[i])
        attention_mask, token_type_ids = None, None
        if "bert" in config.model_name:
            text_dict = self.tokenizer.encode_plus(
                text,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_length,  # Pad & truncate all sentences.
                return_attention_mask=True,  # Construct attn. masks.
                # return_tensors='pt',     # Return pytorch tensors.
            )
            input_ids, attention_mask, token_type_ids = \
                text_dict['input_ids'], text_dict['attention_mask'], text_dict['token_type_ids']
            input = self.tokenizer.convert_ids_to_tokens(list(input_ids))
            seq_len = len(input)
        else:
            # 如果是cnn rnn， transformer则使用自建的dictionary 来处理
            text = text.split()
            text = text if len(text) < self.max_length else text[:self.max_length]
            input_ids = [self.tokenizer.indexer('<SOS>')] + \
                [self.tokenizer.indexer(x) for x in text] + \
                    [self.tokenizer.indexer('<EOS>')]
            input = ['<SOS>'] + text + ['<EOS>']
            seq_len = len(input)
            
        output = {
            "input": input,
            "token_ids": input_ids,
            'attention_mask': attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label,
            "seq_len": seq_len,
        }
        return output
            

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad 添加 0
        """
        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    seq_len = [data["seq_len"] for data in batch]
    max_length = max(seq_len)
    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    labels = torch.tensor([data["labels"] for data in batch])
    token_ids_padded = padding(token_ids, max_length)
    if config.model_name == "bert":
        token_type_ids_padded = padding(token_type_ids, max_length)
        attention_mask_padded = padding(attention_mask, max_length)
        return token_ids_padded, attention_mask_padded, token_type_ids_padded, labels
    else:
        return token_ids_padded, labels
        

# if __name__ == "__main__":
#     tokenizer = BertTokenizer.from_pretrained("/Volumes/yyx/projects/Chinese-Text-Classification/DL/pretrained_model/roberta_wwm_large_ext")
#     # model = BertModel.from_pretrained("/Volumes/yyx/projects/Chinese-Text-Classification/DL/pretrained_model/roberta_wwm_large_ext")
#     train_dataset = MedicalData(config.train_data_file, tokenizer=tokenizer, char_level=True)
#     print(train_dataset.data["text"].values.tolist()[1])
#     print(train_dataset[1])
