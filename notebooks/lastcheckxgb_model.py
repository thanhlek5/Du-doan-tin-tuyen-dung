import os 
import sys 
project_root = os.path.abspath(os.path.join(os.getcwd(),'..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.model_utils import trainModel, tuneModel, saveModel,save_params_to_json
from src.preprocessor_utils import split_data
import pandas as pd 
from src.preprocessor_utils import get_dataCount, get_datatfidf,get_dataword2vec, trans_data
from src.eval_metric import Metric
from src.pipeline_imbalanced import smote_pipeline, smote_under_pipeline
# path configs 
path_test = os.path.join(project_root,'fraud-detection-post',"data","data_test.csv")
df = pd.read_csv(path_test)
x_test,y_test = split_data(df)
X_test = trans_data(x_test,"count")

# ----------------------------huấn luyênj với vec hóa bằng countvector 
x_train, y_train = get_dataCount()
x_su,y_su = smote_under_pipeline(x_train,y_train)
x_sm,y_sm = smote_pipeline(x_train,y_train)
model_count = trainModel(x_train,y_train,"xgb")
model_csm = trainModel(x_sm,y_sm,"xgb")
model_csu = trainModel(x_su,y_su,"xgb")

eval_count = Metric(model_count,X_test,y_test)
eval_sm = Metric(model_csm,X_test,y_test)
eval_su = Metric(model_csu,X_test,y_test)

eval_count.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#          0       0.99      1.00      0.99      3403
#         1       0.92      0.81      0.86       173
#
#    accuracy                           0.99      3576
#   macro avg       0.95      0.90      0.93      3576
#weighted avg       0.99      0.99      0.99      3576

#ROC-AUC: 0.9945
#PR-AUC (AUPRC): 0.9488 (Quan trọng cho Fraud)
eval_sm.evaluate_model("xgb",0.2)
#--- ĐÁNH GIÁ: XGB (Threshold=0.2) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.99      0.99      3403
#           1       0.82      0.84      0.83       173
#
#    accuracy                           0.98      3576
#   macro avg       0.91      0.92      0.91      3576
#weighted avg       0.98      0.98      0.98      3576

#ROC-AUC: 0.9942
#PR-AUC (AUPRC): 0.9264 (Quan trọng cho Fraud)
eval_su.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.98      0.99      3403
#           1       0.72      0.90      0.80       173
#
#    accuracy                           0.98      3576
#   macro avg       0.86      0.94      0.89      3576
#weighted avg       0.98      0.98      0.98      3576

#ROC-AUC: 0.9915
#PR-AUC (AUPRC): 0.9171 (Quan trọng cho Fraud)
#
#-------------Tune model---------------
print(model_count.get_params())
param_rcv = { 
    "n_estimators": [50,100,300,500],
    "learning_rate": [0.01,0.05,0.1],
    "max_depth":[3,5,10,15],
    "subsmaple":[0.5,0.7,1.0],
    "colsmaple_bytree":[0.5,0.7,1.0],
    "min_child_weight":[1,3,5,7],
    "scale_pos_weight":[1,30,50,70,100]
    }
tune_count = tuneModel(x_train,y_train,"xgb",param_rcv,"rcv",n_iter = 50)
eval_tc =Metric(tune_count[0],X_test,y_test)
eval_tc.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.99      0.99      3403
#           1       0.87      0.90      0.88       173
#
#    accuracy                           0.99      3576
#   macro avg       0.93      0.94      0.94      3576
#weighted avg       0.99      0.99      0.99      3576
#
#ROC-AUC: 0.9942
#PR-AUC (AUPRC): 0.9496 (Quan trọng cho Fraud)
print(tune_count[1])
param_gcv = {
    "n_estimators": [500],
    "learning_rate":[0.1],
    "max_depth":[3,5,7],
    "min_child_weight": [3,5,7],
    "scale_pos_weight":[1]
    }
tune_gcvc =tuneModel(x_train,y_train,"xgb",param_gcv,"gcv")
eval_gcvc = Metric(tune_gcvc[0],X_test,y_test)
eval_gcvc.evaluate_model("xgb",0.3)
#-- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.99      0.99      3403
#           1       0.84      0.89      0.87       173
#
#    accuracy                           0.99      3576
#   macro avg       0.92      0.94      0.93      3576
#weighted avg       0.99      0.99      0.99      3576

#ROC-AUC: 0.9941
#PR-AUC (AUPRC): 0.9473 (Quan trọng cho Fraud)
print(tune_gcvc[1])
path_save_md = os.path.join(project_root,"fraud-detection-post","models","model_xgb_tunec_03.pkl")
saveModel(tune_count[0],path_save_md)
path_pms_tunec = os.path.join(project_root,"fraud-detection-post","configs","param_xgb_tunec_03.json")
save_params_to_json(tune_count[1],path_pms_tunec)
#-----------------------Thử với tfidf---------------------
x_tf,y_tf = get_datatfidf()
x_tsm,y_tsm = smote_pipeline(x_tf,y_tf) 
x_tsu,y_tsu = smote_under_pipeline(x_tf,y_tf)
X_testtf = trans_data(x_test,'tfidf')





model_tf = trainModel(x_tf,y_tf,"xgb")
model_tsm = trainModel(x_tsm,y_tsm,"xgb")
model_tsu = trainModel(x_tsu,y_tsu,"xgb")
eval_tf = Metric(model_tf,X_testtf,y_test)
eval_tsm = Metric(model_tsm,X_testtf,y_test)
eval_tsu = Metric(model_tsu,X_testtf,y_test)

eval_tf.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#           0       0.99      1.00      0.99      3403
#           1       0.94      0.80      0.86       173
#
#    accuracy                           0.99      3576
#   macro avg       0.96      0.90      0.93      3576
#weighted avg       0.99      0.99      0.99      3576
#
#ROC-AUC: 0.9919
#PR-AUC (AUPRC): 0.9284 (Quan trọng cho Fraud)

eval_tsm.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.99      0.99      3403
#           1       0.87      0.79      0.83       173
#
#    accuracy                           0.98      3576
#   macro avg       0.93      0.89      0.91      3576
#weighted avg       0.98      0.98      0.98      3576
#
#ROC-AUC: 0.9889
#PR-AUC (AUPRC): 0.9130 (Quan trọng cho Fraud)

eval_tsu.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#          0       0.99      0.98      0.99      3403
#           1       0.68      0.88      0.77       173
#
#    accuracy                           0.97      3576
#   macro avg       0.84      0.93      0.88      3576
#weighted avg       0.98      0.97      0.98      3576
#
#ROC-AUC: 0.9882
#PR-AUC (AUPRC): 0.8949 (Quan trọng cho Fraud)

#--------------------- tune---------------
tune_tfr = tuneModel(x_tf,y_tf,"xgb",param_rcv,"rcv",n_iter =50)
eval_tfr = Metric(tune_tfr[0],X_testtf,y_test)
eval_tfr.evaluate_model("xgb",0.5)
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.99      1.00      0.99      3403
#           1       0.94      0.72      0.81       173
#
#    accuracy                           0.98      3576
#   macro avg       0.96      0.86      0.90      3576
#weighted avg       0.98      0.98      0.98      3576

#ROC-AUC: 0.9915
#PR-AUC (AUPRC): 0.9221 (Quan trọng cho Fraud)
print(tune_tfr[1])


tune_tfg = tuneModel(x_tf,y_tf,"xgb",param_gcv,"gcv")
eval_tfg = Metric(tune_tfg[0],X_testtf,y_test)
eval_tfg.evaluate_model("xgb",0.5)
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#             precision    recall  f1-score   support
#
#           0       0.99      1.00      0.99      3403
#           1       0.95      0.71      0.81       173
#
#    accuracy                           0.98      3576
#   macro avg       0.97      0.85      0.90      3576
#weighted avg       0.98      0.98      0.98      3576
#
#ROC-AUC: 0.9922
#PR-AUC (AUPRC): 0.9263 (Quan trọng cho Fraud)
print(tune_tfg[1])

#-----------------------word2vec---------------------
x_2v,y_2v = get_dataword2vec()
x_te2v = trans_data(x_test,"word2vec")

model_2v = trainModel(x_2v,y_2v,"xgb")
eval_2v = Metric(model_2v,x_te2v,y_test)
eval_2v.evaluate_model("xgb")
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.98      1.00      0.99      3403
#           1       0.96      0.60      0.74       173
#
#    accuracy                           0.98      3576
#   macro avg       0.97      0.80      0.86      3576
#weighted avg       0.98      0.98      0.98      3576
#
#ROC-AUC: 0.9723
#PR-AUC (AUPRC): 0.8576 (Quan trọng cho Fraud)

x_2sm,y_2sm = smote_pipeline(x_2v,y_2v)
x_2su,y_2su = smote_under_pipeline(x_2v,y_2v)

model_2sm = trainModel(x_2sm,y_2sm,"xgb")
eval_2sm = Metric(model_2sm,x_te2v,y_test)
eval_2sm.evaluate_model("xgb")
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.99      0.99      3403
#           1       0.85      0.73      0.79       173
#
#    accuracy                           0.98      3576
#   macro avg       0.92      0.86      0.89      3576
#weighted avg       0.98      0.98      0.98      3576
#
#ROC-AUC: 0.9691
#PR-AUC (AUPRC): 0.8459 (Quan trọng cho Fraud)

model_2su = trainModel(x_2su,y_2su,"xgb")
eval_2su = Metric(model_2su,x_te2v,y_test)
eval_2su.evaluate_model("xgb")
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.97      0.98      3403
#           1       0.59      0.82      0.69       173
#
#    accuracy                           0.96      3576
#   macro avg       0.79      0.89      0.83      3576
#weighted avg       0.97      0.96      0.97      3576
#
#ROC-AUC: 0.9695
#PR-AUC (AUPRC): 0.8201 (Quan trọng cho Fraud)

tune_2vr = tuneModel(x_2v,y_2v,"xgb",param_rcv,"rcv",n_iter = 50)
eval_tune2r = Metric(tune_2vr[0],x_te2v,y_test)
eval_tune2r.evaluate_model("xgb")
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.99      1.00      0.99      3403
#           1       0.90      0.71      0.79       173
#
#    accuracy                           0.98      3576
#   macro avg       0.94      0.85      0.89      3576
#weighted avg       0.98      0.98      0.98      3576
#
#ROC-AUC: 0.9750
#PR-AUC (AUPRC): 0.8652 (Quan trọng cho Fraud)
print(tune_2vr[1])
param_gcv = {
    "n_estimators": [500],
    "learning_rate":[0.1],
    "max_depth":[8,10,12],
    "min_child_weight": [3,5,7],
    "scale_pos_weight":[10,30,50]
    }
tune_2vg = tuneModel(x_2v,y_2v,"xgb",param_gcv,"gcv")
eval_tune2g = Metric(tune_2vg[0],x_te2v,y_test)
eval_tune2g.evaluate_model("xgb")
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.99      1.00      0.99      3403
#           1       0.90      0.72      0.80       173
#
#    accuracy                           0.98      3576
#   macro avg       0.94      0.86      0.89      3576
#weighted avg       0.98      0.98      0.98      3576
#
#ROC-AUC: 0.9756
#PR-AUC (AUPRC): 0.8662 (Quan trọng cho Fraud)
param_opt = {
            "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 100},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
            "min_child_weight": {"type": "int", "low": 1, "high": 7},
            "scale_pos_weight": {"type": "float", "low": 1, "high": 100, "log": True} 
            }   
tune_2vo = tuneModel(x_2v,y_2v,"xgb",param_opt,"opt",n_trials = 10)
eval_tune2o = Metric(tune_2vo[0],x_te2v,y_test)
eval_tune2o.evaluate_model("xgb")
#--- ĐÁNH GIÁ: XGB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.99      1.00      0.99      3403
#           1       0.92      0.72      0.81       173
#
#    accuracy                           0.98      3576
#   macro avg       0.95      0.86      0.90      3576
#weighted avg       0.98      0.98      0.98      3576
#
#ROC-AUC: 0.9721
#PR-AUC (AUPRC): 0.8605 (Quan trọng cho Fraud)
path_tf = os.path.join(project_root,"fraud-detection-post","models","model_xgb_tf_tuneg.pkl")
saveModel(tune_tfg[0],path_tf)
path_tf = os.path.join(project_root,"fraud-detection-post","configs","param_xgb_tf_tuneg.pkl")
save_params_to_json(tune_tfg[1],path_tf)