import os 
import sys 
project_root = os.path.dirname(os.path.dirname(__file__))
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


# path configs
path_test = os.path.join(project_root,"data","data_train.csv")
path_pre_data_tf = os.path.join(project_root, "models","Preprocess_tfidf_data.pkl")
path_pre_pipe_tf = os.path.join(project_root,"models","Preprocess_tfidf_pipeline.pkl")
path_config = os.path.join(project_root,"configs","lr_rcv.json")

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
