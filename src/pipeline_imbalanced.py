from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import numpy as np
from typing import Any
def smote_pipeline(x:Any,y:pd.Series):
    smote = SMOTE(sampling_strategy="auto",random_state=42)
    x_res,y_res =smote.fit_resample(x,y)
    return x_res,y_res

def smote_under_pipeline(x:Any,y:pd.Series):
    smote =SMOTE(sampling_strategy=0.23,random_state= 42)
    
    undersample = RandomUnderSampler(sampling_strategy=0.93,random_state= 42)
    pipeline = ImbPipeline([
        ('smote',smote),
        ('under',undersample)
    ])
    x_res,y_res = pipeline.fit_resample(x,y)
    return x_res,y_res

