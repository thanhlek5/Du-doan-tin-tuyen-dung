from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import numpy as np

def smote_pipeline(x:pd.DataFrame,y:pd.DataFrame):
    smote = SMOTE(sampling_strategy="auto",random_state=42)
    x_res,y_res =smote.fit_resample(x,y)
    return x_res,y_res

def smote_under_pipeline(x,y):
    smote =SMOTE(sampling_strategy=0.2,random_state= 42)
    
    undersample = RandomUnderSampler(sampling_strategy=0.5,random_state= 42)
    pipeline = ImbPipeline([
        ('smote',smote),
        ('under',undersample)
    ])
    x_rex,y_res = pipeline.fit_resample(x,y)
    return x_rex,y_res

