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
    
    

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def split_data(df:pd.DataFrame) ->tuple:
    x= df.drop("fraudulent",axis= 1)
    y = df['fraudulent']
    return x,y

def load_preprocessor(x) -> Any:
    path_pre = os.path.join(project_root,"models","preprocessor.pkl")
    preprocessor = joblib.load(path_pre)
    return preprocessor.transform(x)

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
    tokens = [word for word in tokens if word not in stop_words]
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