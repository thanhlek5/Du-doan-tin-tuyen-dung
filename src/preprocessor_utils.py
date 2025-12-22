import numpy as np 
import pandas as pd 
import joblib 
import os 
import sys 
from typing import Any

project_root = os.path.abspath(os.path.join(os.getcwd(),".."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
    
    
def split_data(df:pd.DataFrame) ->tuple:
    x= df.drop("fraudulent",axis= 1)
    y = df['fraudulent']
    return x,y

def load_preprocess(path:str) -> Any:
    return joblib.load(path)


def get_dataCount():
    path = os.path.join(project_root,"models","Preprocess_count_data.pkl")
    if not os.path.exists(path):
        print(f"this path does not exists { path}")
    pre = joblib.load(path)
    return pre[0],pre[1]

def get_datatfidf():
    path = os.path.join(project_root,"models","Preprocess_tfidf_data.pkl")
    if not os.path.exists(path):
        print(f"this path does not exists { path}")
    pre = joblib.load(path)
    return pre[0],pre[1]
    
    
def get_dataword2vec():
    path = os.path.join(project_root,"models","Preprocess_word2vec_data.pkl")
    if not os.path.exists(path):
        print(f"this path does not exists { path}")
    pre = joblib.load(path)
    return pre[0],pre[1]

def trans_data(x,name):
    name = name.lower()
    if name == "count":
        path = os.path.join(project_root,"models","Preprocess_count_pipeline.pkl") 
    if name == "tfidf":
        path = os.path.join(project_root,"models","Preprocess_tfidf_pipeline.pkl")  
    if name == "word2vec":
        path = os.path.join(project_root,"models","Preprocess_word2vec_pipeline.pkl")  
    pre = joblib.load(path)
    return pre.transform(x)

# ,'fraud-detection-post'