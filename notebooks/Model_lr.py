import os 
import sys 
project_root = os.path.abspath(os.path.join(os.getcwd(),".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.model_utils import trainModel, tuneModel, saveModel, save_params_to_json
from src.preprocessor_utils import split_data, load_preprocess, transform_preprocessor 
from src.eval_metric import Metric
import pandas as pd 
import numpy as np
from src.pipeline_imbalanced import smote_pipeline, smote_under_pipeline
import json
import shap
import matplotlib.pyplot as plt
from sklearn.base import clone
import joblib

# path configs
path_test = os.path.join(project_root,"fraud-detection-post","data","data_test.csv")
path_pre_data_tf = os.path.join(project_root,"fraud-detection-post", "models","Preprocess_tfidf_data.pkl")
path_pre_pipe_tf = os.path.join(project_root,"fraud-detection-post","models","Preprocess_tfidf_pipeline.pkl")
path_config = os.path.join(project_root,"fraud-detection-post","configs","lr_rcv.json")

with open(path_config, 'r', encoding="utf-8") as f:
    param = json.load(f)

# split and setup data
df_test = pd.read_csv(path_test)
x_test,y_test = split_data(df_test)
pre_data = load_preprocess(path_pre_data_tf)
pre_pipeline = load_preprocess(path_pre_pipe_tf)
X_test = pre_pipeline.transform(x_test)
x_train= pre_data[0]
y_train = pre_data[1]
x_smote ,y_smote = smote_pipeline(x_train, y_train)
x_mutil,y_mutil = smote_under_pipeline(x_train, y_train)


# train model logistic regressor 

model_noim = trainModel(x_train, y_train, model_name = "lr")

eval_noim = Metric(model_noim, X_test, y_test)

eval_noim.evaluate_model("lr",0.25) # PR-AUC (AUPRC): 0.8954

model_smote = trainModel(x_smote, y_smote, "lr")
eval_smote = Metric(model_smote, X_test, y_test)

eval_smote.evaluate_model("lr",0.6) # PR-AUC (AUPRC): 0.9647

model_mutil = trainModel(x_mutil, y_mutil, "lr")

eval_mutil = Metric(model_mutil, X_test, y_test)
eval_mutil.evaluate_model("lr",0.7)


model_tune = tuneModel(x_mutil, y_mutil,"lr",param,cv= 10)

eval_tune = Metric(model_tune[0],X_test,y_test)
eval_tune.evaluate_model("lr",0.9) # -> model hình tốt nhất -> beeswarm _feature 2889 -> từ miss
# từ mis trong missing là từ đc điền vào chỗ trống trong cột dạng text 
masker = shap.maskers.Independent(data=x_train)
explainer = shap.LinearExplainer(model_mutil, masker=masker)
shap_values = explainer(X_test)
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=15)


y_shuffled = np.random.permutation(y_smote)
model_tune_clone = clone(model_tune[0])
model_tune_clone.fit(x_smote,y_shuffled)
eval_clone = Metric(model_tune_clone, X_test, y_test)
eval_clone.evaluate_model("lr",0.5)


print(list(pre_pipeline.named_steps.keys()))

# Lấy đối tượng ColumnTransformer ra
ct = pre_pipeline['preprocessor']

feature_list = []

print("--- ĐANG QUÉT CẤU TRÚC BÊN TRONG ---")
# Duyệt qua từng nhánh xử lý bên trong ColumnTransformer
for name, transformer, columns in ct.transformers_:
    print(f"Đang kiểm tra nhánh: '{name}'...")
    
    try:
        # Trường hợp 1: Nhánh này là một Pipeline con (Ví dụ: cat_pipeline)
        if hasattr(transformer, 'steps'):
            # Lấy bước cuối cùng của nhánh này (thường là OneHotEncoder)
            # để né bước 'cleaner' ở đầu gây lỗi
            names = transformer[-1].get_feature_names_out()
            feature_list.extend(names)
            print(f" -> ✅ Đã lấy được {len(names)} features.")
            
        # Trường hợp 2: Nhánh này là Transformer đơn lẻ
        elif hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out()
            feature_list.extend(names)
            print(f" -> ✅ Đã lấy được {len(names)} features.")
            
        else:
            print(" -> ⚠️ Bỏ qua (Không hỗ trợ lấy tên)")
            
    except Exception as e:
        print(f" -> ❌ Lỗi ở nhánh này: {e}")

print("-" * 30)
print(f"Tổng số features thu thập được: {len(feature_list)}")

# IN RA KẾT QUẢ CUỐI CÙNG
if len(feature_list) > 2889:
    print(f"\n😈 THỦ PHẠM Feature 2889 LÀ: {feature_list[2889]}")
else:
    print(f"\n⚠️ Vẫn chưa đủ số lượng feature (Tìm được {len(feature_list)}, cần > 2889).")
    print("Có thể feature này sinh ra từ bước 'engineer' ở đầu mà ta đã bỏ qua.")
    
path_model = os.path.join(project_root,"models","models_lr.pkl")
saveModel(model_tune[0], path_model)

# _________________20/12_____________________

# -------------------------- train in count, word2vec---------------
path_count = os.path.join(project_root,"fraud-detection-post","models","Preprocess_count_data.pkl")
path_count_pipe = os.path.join(project_root,"fraud-detection-post","models","Preprocess_count_pipeline.pkl")
path_word2vec = os.path.join(project_root,"fraud-detection-post","models","Preprocess_word2vec_data.pkl")
path_word2vec_pipe = os.path.join(project_root,"fraud-detection-post","models","Preprocess_word2vec_pipeline.pkl")

pre_count = joblib.load(path_count_pipe)
X_test = pre_count.transform(x_test)
x_count , y_count =  joblib.load(path_count)
xcount_smote,ycount_smote = smote_pipeline(x_count,y_count)
model_count = trainModel(x_count,y_count,"lr")
evl_count = Metric(model_count,X_test,y_test)
evl_count.evaluate_model("lr",0.3) # -> 0.89

# smote 
ml_count_smote = trainModel(xcount_smote,ycount_smote,"lr")
evl_count_smote = Metric(ml_count_smote,X_test,y_test)
evl_count_smote.evaluate_model("lr",0.1) # -> 0.87

# under-smote 
xcount_muti , ycount_muti = smote_under_pipeline(x_count,y_count)
model_count_muti = trainModel(xcount_muti,ycount_muti,"lr")
evl_muti = Metric(model_count_muti,X_test,y_test)
evl_muti.evaluate_model("lr",0.5) # -> 0.84

# -----------------------------------word2vec--------------------
x_2,y_2 = joblib.load(path_word2vec)
pre_2vec = joblib.load(path_word2vec_pipe)
X_test = pre_2vec.transform(x_test)

ml_2vec = trainModel(x_2,y_2,"lr")
evl_muti = Metric(ml_2vec,X_test,y_test)
evl_muti.evaluate_model("lr",0.5) # -> 0.54

x_2_smote , y_2_smote = smote_pipeline(x_2,y_2)
ml_2sm = trainModel(x_2_smote,y_2_smote,"lr")
evl_muti = Metric(ml_2sm,X_test,y_test)
evl_muti.evaluate_model("lr",0.5) # 0.41

x_2_muti, y_2_muti = smote_under_pipeline(x_2,y_2)
ml_2muti = trainModel(x_2_muti,y_2_muti,"lr")
evl_muti = Metric(ml_2muti,X_test,y_test)
evl_muti.evaluate_model("lr",0.5) # 0.43

# dùng count và smmote  cho ra khả năng dự đoán tốt nhất giờ ta sẽ tune model để xem thử 
param_rcv = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
    } 
model_tune_count = tuneModel(xcount_smote,ycount_smote,"lr",param_rcv,"rcv",n_iter = 50)
eval_tune_count_rcv = Metric(model_tune_count[0], X_test,y_test)
eval_tune_count_rcv.evaluate_model("lr",0.5)
print(model_tune_count[1])
param_otp = { 
    "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
    "penalty": ["l1", "l2"],
      "solver": ["saga"]
    }
model_tune_count_otp = tuneModel(xcount_smote,ycount_smote,"lr",param_otp,"opt",n_trials = 50)
evl_opt = Metric(model_tune_count_otp[0],X_test,y_test)
evl_opt.evaluate_model("lr",0.3) # -> 0.88 
# lưu mô hình cao nhất của count 
path_model = os.path.join(project_root,"fraud-detection-post","models","model_lr_count.pkl")
saveModel(model_count,path_model) 


