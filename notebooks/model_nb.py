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

path_test = os.path.join(project_root,'fraud-detection-post',"data","data_test.csv")
df = pd.read_csv(path_test)
x_test,y_test = split_data(df)
X_test = trans_data(x_test,"count")
#-------=
x_train, y_train = get_dataCount()
x_su,y_su = smote_under_pipeline(x_train,y_train)
x_sm,y_sm = smote_pipeline(x_train,y_train)

model_count = trainModel(x_train,y_train,"nb")
model_csm = trainModel(x_sm,y_sm,"nb")
model_csu = trainModel(x_su,y_su,"nb")

eval_count = Metric(model_count,X_test,y_test)
eval_sm = Metric(model_csm,X_test,y_test)
eval_su = Metric(model_csu,X_test,y_test)

eval_count.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.92      0.96      3403
#           1       0.36      0.88      0.51       173
#
#    accuracy                           0.92      3576
#   macro avg       0.68      0.90      0.73      3576
#weighted avg       0.96      0.92      0.93      3576
#
#ROC-AUC: 0.9661
#PR-AUC (AUPRC): 0.6052 (Quan trọng cho Fraud)

eval_sm.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.95      0.97      3403
#           1       0.48      0.88      0.62       173
#
#    accuracy                           0.95      3576
#   macro avg       0.74      0.91      0.80      3576
#weighted avg       0.97      0.95      0.95      3576

#ROC-AUC: 0.9717
#PR-AUC (AUPRC): 0.6870 (Quan trọng cho Fraud)
eval_su.evaluate_model("xgb",0.3)
#--- ĐÁNH GIÁ: XGB (Threshold=0.3) ---
#              precision    recall  f1-score   support
#
#          0       0.99      0.94      0.96      3403
#           1       0.41      0.88      0.56       173
#
#   accuracy                           0.93      3576
#   macro avg       0.70      0.91      0.76      3576
#weighted avg       0.97      0.93      0.94      3576

#ROC-AUC: 0.9699
#PR-AUC (AUPRC): 0.6669 (Quan trọng cho Fraud)

param_opt = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],  # 1.0 là mặc định
    'fit_prior': [True, False]
}
tune_opt = tuneModel(x_train,y_train,"nb",param_opt,"opt",n_trials = 50)
eval_optc = Metric(tune_opt[0],X_test,y_test)
eval_optc.evaluate_model("nb") 

#--- ĐÁNH GIÁ: NB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.99      0.96      0.98      3403
#           1       0.52      0.86      0.65       173
#
#    accuracy                           0.95      3576
#   macro avg       0.75      0.91      0.81      3576
#weighted avg       0.97      0.95      0.96      3576
#
#ROC-AUC: 0.9738
#PR-AUC (AUPRC): 0.7023 (Quan trọng cho Fraud)

#-----------------------tf----------------
x_tf,y_tf = get_datatfidf()
X_tetf = trans_data(x_test,"tfidf")
model_tf = trainModel(x_tf,y_tf,"nb")
eval_ = Metric(model_tf,X_tetf,y_test)
eval_.evaluate_model("nb") 
#--- ĐÁNH GIÁ: NB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.97      1.00      0.98      3403
#           1       0.98      0.35      0.52       173
#
#    accuracy                           0.97      3576
#   macro avg       0.98      0.68      0.75      3576
#weighted avg       0.97      0.97      0.96      3576
#
#ROC-AUC: 0.9361
#PR-AUC (AUPRC): 0.7061 (Quan trọng cho Fraud)

#-------------------tune--------------- 
tune_tf = tuneModel(x_tf,y_tf,"nb",param_opt,"opt",n_trials =50)
eval_optf = Metric(tune_tf[0],X_tetf,y_test)
eval_optf.evaluate_model('nb')
#--- ĐÁNH GIÁ: NB (Threshold=0.5) ---
#              precision    recall  f1-score   support
#
#           0       0.98      1.00      0.99      3403
#           1       0.92      0.57      0.70       173
#
#    accuracy                           0.98      3576
#   macro avg       0.95      0.78      0.84      3576
#weighted avg       0.98      0.98      0.97      3576
#
#ROC-AUC: 0.9774
#PR-AUC (AUPRC): 0.8152 (Quan trọng cho Fraud)

path_save_md = os.path.join(project_root,"fraud-detection-post","models","model_nb_tunef_05.pkl")
saveModel(tune_tf[0],path_save_md)
path_pms_tunef = os.path.join(project_root,"fraud-detection-post","configs","param_nb_tunef_03.json")
save_params_to_json(tune_tf[1],path_pms_tunef)