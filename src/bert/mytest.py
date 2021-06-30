# -*- coding: utf-8 -*-


!pip install torch
!pip install pytorch_transformers
!pip install transformers
!pip install IProgress


from __future__ import absolute_import, division, print_function

import json
import os

import torch
import torch.nn.functional as F
from nltk import word_tokenize
from pytorch_transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)



class BertNer(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits

class Ner:

    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        model = BertNer.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config


!pip install ipywidgets
!pip install tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from ipywidgets import IntProgress

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")



model = BertNer.from_pretrained('/Users/Pablo/Downloads/pytorch')











