#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
from datasets import DatasetDict, Dataset
from utlis import *


# In[2]:


def rm_special_keys(review):
    special_character = re.compile("�+")
    return special_character.sub(r'', review)

def rm_punctuation(review):
    punctuation = re.compile(r"[!#$%&()*+;<=>?@[\]^_`{|}~]+")
    return punctuation.sub(r"", review)

def rm_emoji(review):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        u"\U0001F700-\U0001F77F"  # Alchemical Symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F004-\U0001F0CF"  # Mahjong Tiles
        u"\U0001F170-\U0001F251"  # Enclosed Characters
        u"\U0001F300-\U0001F9F9"  # Additional symbols and emojis
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', review)
    return text

def rm_urls_paths(text):
    # Define a regex pattern to match both URLs and file paths
    url_pattern = r'https?[:]//\S+|www\.\S+'
    path_pattern = r'(?:(?:[a-z]:\\|\\\\|/)[^\s|/]+(?:/[^\s|/]+)*)'
    combined_pattern = f'({url_pattern})|({path_pattern})'
    cleaned_text = re.sub(combined_pattern, '', text)
    return cleaned_text

def normalize_annotatation(text):
    khach_san = "\bkhach san ?|\bksan ?|\bks ?"
    return re.sub("\bnv ?", "nhân viên",re.sub(khach_san, "khách sạn", text))

def rm_escape_characters(text):
    cleaned_text = text.replace('\r', '').replace('\n', '').replace('\t', '').replace('\q', '').replace('\w', '').replace('\s', '')
    return cleaned_text

def clean_text(review):
    cleaned_review = {"Review": rm_escape_characters(normalize_annotatation(rm_special_keys(rm_punctuation(rm_emoji(rm_urls_paths(review['Review'].lower()))))))}
    return cleaned_review


# In[3]:


class preprocess():
    def __init__(self):
        self.proj_path = get_proj_path()
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.segmenter = VnCoreNLP(os.path.join(self.proj_path, 'vncorenlp', 'VnCoreNLP-1.1.1.jar'), annotators="wseg", max_heap_size='-Xmx500m')
        self.feature = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']
        
    def segment(self, df):
        return {"Segment": " ".join([" ".join(sen) for sen in self.segmenter.tokenize(df["Review"])])}
        
    def tokenize(self, df):
        return self.tokenizer(df["Segment"], truncation=True, padding=True, max_length=165)
    
    def label(self, example):
        return {'labels_regressor': np.array([example[i] for i in self.feature]),
            'labels_classifier': np.array([int(example[i] != 0) for i in self.feature])}
    
    def rm_stopwords(self, text, remove_stopwords=True):
        stopword_path = os.path.join(self.proj_path, "vn_stopwords", "vietnamese-stopwords-dash.txt")
        with open(stopword_path, 'r', encoding='utf-8') as file:
            stop_words = set(file.read().splitlines())    
        words = text['Review'].split()
        if remove_stopwords:
            words = [word for word in words if word.lower() not in stop_words]
        cleaned_text = ' '.join(words)
        return {"Review": cleaned_text}
        
    def run(self, dataset):
        dataset = dataset.map(clean_text)
        dataset = dataset.map(self.segment)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset = dataset.map(self.label)
        dataset = dataset.map(self.rm_stopwords)
        dataset.set_format("torch")
        
        return dataset