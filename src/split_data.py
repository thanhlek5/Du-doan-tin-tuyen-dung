import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import sys 
project_root = os.path.abspath(os.path.join(os.getcwd(),".."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
path_data = os.path.join(project_root,"fraud-detection-post","data","fake_job_postings.csv")
path_train =os.path.join(project_root,"fraud-detection-post","data","data_train.csv") 
path_test =os.path.join(project_root,"fraud-detection-post","data","data_test.csv")

df = pd.read_csv(path_data)
x = df.drop('fraudulent', axis= 1)
y = df['fraudulent']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 42,stratify=y)

df_train = pd.DataFrame(x_train).reset_index(drop=True)
df_test = pd.DataFrame(x_test).reset_index(drop=True)

df_train['fraudulent'] = y_train.reset_index(drop = True)
df_test['fraudulent'] = y_test.reset_index(drop = True)

df_train.to_csv(path_train, index= False)
df_test.to_csv(path_test,index= False)
