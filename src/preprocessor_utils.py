import numpy as np 
import pandas as pd 
import joblib 
import os 
import sys 
from typing import Any
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load("en_core_web_sm")
project_root = os.path.abspath(os.path.join(os.getcwd(),".."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
    

STOPWORDS = set(spacy.lang.en.stop_words.STOP_WORDS)

def split_data(df:pd.DataFrame) ->tuple:
    x= df.drop("fraudulent",axis= 1)
    y = df['fraudulent']
    return x,y

def load_preprocessor(x) -> Any:
    path_pre = os.path.join(project_root,"models","preprocessor.pkl")
    preprocessor = joblib.load(path_pre)
    return preprocessor.transform(x)

#  lấy được ở phần analysis
whitelist = {'show', 'unless', 'me', 'anywhere', 'he', 'again', 'from', 'my', 'may', 'before', 'full', 'name', 'done', 'nothing', 'others', 'per', 'above', 'below', 'six', 'your', 'down', 'own', 'hence', 'thereby', 'within', 'call', 'ours', 'third', 'must', 'off', 'say', 'ten', 'eight', 'his', 'should', 'serious', 'any', 'otherwise', 'mostly', 'much', 'several', 'under', 'no', 'amount', 'toward', 'amongst', 'via', 'mine', 'hundred', 'whose'}
final_stopwords = STOPWORDS - whitelist


def clear_text(text:str) ->str:
    """
    làm sạch văn bảng 
    """
    if not isinstance(text,str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>',"",text)
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in final_stopwords]
    clear_text = " ".join(tokens)
    return clear_text

def normalize_text(text):
    """
    Chuẩn hóa văn bản
    """
    doc = nlp(text)
    normalized_words = [token.lemma_ for token in doc]
    normalized_text = ' '.join(normalized_words)
    return normalized_text