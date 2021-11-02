#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:32:28 2021

@author: Pablo
"""


from transformers import pipeline

model_path_or_name = "mrm8488/bert-spanish-cased-finetuned-ner"


model_path_or_name = '/Users/Pablo/Downloads/Elvi_exp/Ib2b'

nlp_ner = pipeline(
    "ner",
    model=model_path_or_name,
    tokenizer=(
        model_path_or_name,  
        {"use_fast": False}
))


text = 'I am suffering a disease of  head tumour, others treatments have been aquired'

nlp_ner(text)