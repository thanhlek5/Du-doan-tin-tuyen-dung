import pandas as pd 
import numpy as np  
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
import optuna 
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna_integration import OptunaSearchCV
import os
import sys 
import traceback
project_root = os.path.abspath(os.path.join(os.getcwd(),".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from typing import Any 

def parse_optuna_params(json_params: dict) -> dict:
    """
    Chuyển đổi dictionary từ JSON sang Optuna Distributions.
    Quy ước JSON:
    - Số thực: {"type": "float", "low": 0.01, "high": 100, "log": true}
    - Số nguyên: {"type": "int", "low": 1, "high": 100}
    - Danh sách thường: ["l1", "l2"] -> Tự hiểu là Categorical
    """
    new_params = {}
    for key, val in json_params.items():
        # Nếu là dict định nghĩa distribution
        if isinstance(val, dict) and "type" in val:
            dtype = val["type"]
            if dtype == "float":
                new_params[key] = FloatDistribution(val["low"], val["high"], log=val.get("log", False))
            elif dtype == "int":
                new_params[key] = IntDistribution(val["low"], val["high"], step=val.get("step", 1), log=val.get("log", False))
            elif dtype == "categorical":
                new_params[key] = CategoricalDistribution(val["choices"])
        # Nếu là list bình thường -> CategoricalDistribution
        elif isinstance(val, list):
            new_params[key] = CategoricalDistribution(val)
        else:
            new_params[key] = CategoricalDistribution([val])
            
    return new_params


def loadConfig(config_path : str) ->dict:
    """
    Load config
    """
    if not os.path.exists(config_path):
        print(f"Doesn't exist this config: {config_path}. using default params")
        return {}
    try:
        with open(config_path,"r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] -> {e} ,Can not read this file!")
        return {}

def loadModel(path: str) -> any:
    """Load model"""
    if not os.path.exists(path):
        print(f"Doesn't exist this model: {path}")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print("[ERROR] -> {e} ,Can not load this model")
        return None

def saveModel(model: any,model_path : str) -> str:
    """Save model"""
    joblib.dump(model,model_path)
    return "DONE"

def get_model_instance(model_name, params = None) -> Any:
    """
    please chosse one of {rf, lr, xgb, nb}
    """
    if params is None: params = {}
    name = model_name.lower()
    
    if name == "rf": return RandomForestClassifier(**params,random_state=42)
    if name == "lr": return LogisticRegression(**params,random_state=42,max_iter=100)
    if name == "xgb": return XGBClassifier(**params,random_state = 42)
    if name == "nb": return MultinomialNB(**params)
    
    raise ValueError("the selected model isn't supported")

def trainModel(x_train: pd.DataFrame,y_train: pd.DataFrame,model_name: str, config_path : str = None) -> Any:
    if config_path is None: params = None
    params = loadConfig(config_path = config_path)
    name = model_name.lower()
    try:
        model = get_model_instance(model_name = name, params = params)
        print(f"Training {model_name.upper()}......")
        model.fit(x_train,y_train)
        print("Train successful")
        return model
    except Exception as e:
        print(f"[ERROR]: {e}")
        return None

def fit_hyperparams(model: any,params: dict,score: str = "f1", cv :int = 5, name_search :str = "gcv", **kwargs) -> any:
    name = name_search.lower()
    
    n_jobs = kwargs.get("n_jobs", -1)
    verbose = kwargs.get("verbose", 2)

    if name == "gcv": 
        return GridSearchCV(
            estimator= model,
            param_grid= params,
            cv= cv,
            scoring= score,
            n_jobs= n_jobs,
            verbose= verbose
        )
    elif name == "rcv":
        n_iter = kwargs.get("n_iter", 10)
        return RandomizedSearchCV(
            estimator= model,
            param_distributions= params,
            n_iter= n_iter,
            n_jobs= n_jobs,
            verbose= verbose,
            scoring= score,
            cv= cv
        )
    elif name == "opt":
        n_trials = kwargs.get("n_trials",10)
        timeout = kwargs.get("timeout", None)
        return OptunaSearchCV(
            estimator= model,
            param_distributions= params,
            n_trials= n_trials,
            n_jobs= n_jobs,
            timeout= timeout,
            scoring = score,
            cv = cv,
            verbose = verbose
        )
    else:
        raise ValueError(f"the selected model is not supported")
    
def tuneModel(x_train: pd.DataFrame, y_train: pd.DataFrame, model_name:str,raw_params: dict = None, name_search: str = "gcv", score: str = "f1", cv :int = 5, **kwargs) -> tuple:
    model_name = model_name.lower()
    search_name = name_search.lower()
    
    if raw_params is None: 
        print("config is empty")
        return None,None
    
    if search_name == "opt": params = parse_optuna_params(raw_params)
    elif search_name == "rcv":
        params = {}
        for k,v in raw_params.items():
            params[k] = v if isinstance(v,list) else [v]
    else:
        params = raw_params
    
    try:
        model = get_model_instance(model_name)
    except ValueError as e:
        print(e)
        traceback.print_exc()
        return None,None
    
    print(f"---------Starting Tuning {model_name.upper()}-------------")
    
    try:
        search = fit_hyperparams(model,params,score,cv,name_search,**kwargs)
        
        search.fit(x_train,y_train)
        print("=======result tune============")
        print(f"Best params: {search.best_params_}")
        print(f"Best {score}: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_
    except Exception as e:
        print(f"[ERROR]: {e}")
        return None,None


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def save_params_to_json(params: dict, filepath: str) -> str:
    """
    Lưu dictionary best_params vào file .json
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, cls=NpEncoder, indent=4)
        print(f" Đã lưu params vào: {filepath}")
    except Exception as e:
        print(f" Lỗi khi lưu JSON: {e}")
    
    
    