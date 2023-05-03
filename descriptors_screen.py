

import pandas as pd
import numpy as np
from QSAR_package.feature_preprocess import correlationSelection

def judge_des_type(tr_x):
    if np.array(tr_x).max() == 1 and np.array(tr_x).min() == 0:
        a = True
    else:
        a = False
    return a

def mean_var(tr_x):
    a = list(tr_x.columns)
    _vars = []
    for i in a:
        var = np.var(tr_x[i])
        _vars.append(var)
    dic = dict(zip(a, _vars))
    varresult = [i for i in a if dic[i] > np.mean(_vars)]
    return varresult

def pearson(tr_x,tr_y):
    corr = correlationSelection()
    corr.PearsonXX(tr_x, tr_y,threshold_low = 0.15, threshold_up = 0.85)
    pearsonresult = corr.selected_tr_x.columns.tolist()
    return pearsonresult

def des_screen(tr_x,tr_y):
    a = judge_des_type(tr_x)
    if a:
        des_list = mean_var(tr_x)
    else:
        des_list = pearson(tr_x,tr_y)
    return des_list
    
