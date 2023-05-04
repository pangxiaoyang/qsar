import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score,matthews_corrcoef

#y_pre is the predictive probability of all models
def calDM(y_proba):
    mean = y_proba.mean(axis=1)
    std = y_proba.std(axis=1)
    DM = np.array([])
    for i in range(len(y_proba)):
        normal = norm(mean.iloc[i],std.iloc[i])
        if mean.iloc[i] >= 0.5:
            DM = np.concatenate([DM,np.array([normal.cdf(0.5)])])
        else:
            DM = np.concatenate([DM,np.array([1-normal.cdf(0.5)])])
    return DM

def decode(proba):
    l = []
    for i in range(len(proba)):
        if proba[i] > 0.5:
            l.append(1)
        else:
            l.append(0)
    return l

def show_metrics(y_true, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    se  = float(tp) / ( float(tp)+float(fn) )
    sp  = float(tn) / ( float(tn)+float(fp) )
    accuracy   = accuracy_score(y_true, y_pred)
    MCC        = matthews_corrcoef(y_true, y_pred)
    metrics=[tp,tn,fp,fn,accuracy,MCC,se,sp]
    return metrics


def cal_Threshold(df):
    for i in range(50,len(df)):
        if accuracy_score(decode(df.iloc[:i,0]),df['y_true'][:i]) <=0.9:
            coverage = i/len(df)
            Threshold = df['DM'][i]
            break
    return Threshold,coverage

def cal_performance_AD(df,Threshold):
    AD_index = np.where(df['DM'] <= Threshold)[0]
    result = show_metrics(decode(df.iloc[AD_index,0]),df['y_true'][AD_index])
    return result
    
def cal_AD(tr_proba,tr_y,te_proba,te_y):
    dstd_pro = calDM(tr_proba)
    tr_proba['DM'] = dstd_pro
    tr_proba['y_true'] = tr_y
    dstd_pro = calDM(te_proba)
    te_proba['DM'] = dstd_pro
    te_proba['y_true'] = te_y
    cv5 = tr_proba.sort_values(by = 'DM',ascending = True).reset_index(drop = True,inplace = True)
    
    result = []
    for i in range(len(tr_proba.columns)):
        df_tr = cv5.iloc[:,[i,-2,-1]]
        Threshold,coverage = cal_Threshold(df_tr)
        metrics_tr = cal_performance_AD(df_tr,Threshold)
        df_te = te_proba.iloc[:,[i,-2,-1]]
        metrics_te = cal_performance_AD(df_te,Threshold)
        result_info = {'Threshold':Threshold,'coverage':coverage,
                       'tr_accuracy':metrics_tr[4],'tr_MCC':metrics_tr[5],
                       'te_accuracy':metrics_te[4],'te_MCC':metrics_te[5],
                       'te_se':metrics_te[6],'te_sp':metrics_te[7]}
        result.append(result_info)
    result = pd.DataFrame(result)
    return result
        
