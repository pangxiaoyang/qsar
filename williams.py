import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,make_scorer
from math import sqrt


def show_metrics(y_true, y_pred):
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    metrics = [R2,MAE,RMSE]
    return metrics

def cal_h(tr_x,X):
    tr_arr = tr_x.values
    XTX = tr_arr.T.dot(tr_arr)
    X_arr = X.values
    h = []
    for i in range(len(X_arr)):
        hi = (X_arr[i].T.dot(np.linalg.inv(XTX))).dot(X_arr[i])
        h.append(hi)
    return h

def cal_performance_AD(df,h_):
    index_AD = []
    for i in range(df):
        if df['h'][i] <= h_ and -3 <= (df['y_true'][i]-df['y_pred'][i]) <= 3:
            index_AD.append(i)
    coverage = len(index_AD)/len(df)
    metrics = show_metrics(df['y_true'][index_AD.values], df['y_pred'][index_AD.values])
    metrics.append(coverage)
    return metrics

def cal_williams(tr_x,tr_y,tr_pred,te_x,te_y,te_pred,williams_path):
    tr_h = cal_h(tr_x,tr_x)
    h_ = 3*(len(tr_x.columns)+1)/len(tr_x)
    te_h = cal_h(tr_x,te_x)
    df_tr = pd.DataFrame([tr_h,tr_y,tr_pred],columns = ['h','y_true','y_pred'])
    coverage_tr,metrics_tr = cal_performance_AD(df_tr,h_)
    df_te = pd.DataFrame([te_h,te_y,te_pred],columns = ['h','y_true','y_pred'])
    coverage_te,metrics_te = cal_performance_AD(df_te,h_)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.xlim([0, 0.15])
    plt.ylim([-4, 4])
    plt.plot(tr_h, (df_tr['y_true']-df_tr['y_pred']), 'bo',markerfacecolor='none', label='training set',alpha=0.5,markersize=5)
    plt.plot(te_h, (df_te['y_true']-df_te['y_pred']), 'rx', label='test set',alpha=0.5,markersize=5)
    plt.axvline(h_ ,color='black',linestyle="--", lw=1)
    plt.axhline(-3 ,color='black',linestyle="--", lw=1)
    plt.axhline(3 ,color='black',linestyle="--", lw=1)
    plt.legend(loc='best')
    plt.xlabel("hi")
    plt.ylabel("Standardized residual")
    plt.savefig(williams_path, dpi=600, bbox_inches="tight")
    
    result_dict = {'h_':h_,'tr_coverage':coverage_tr,'tr_R2':metrics_tr[0],'tr_MAE':metrics_tr[1],
                   'tr_RMSE':metrics_tr[2],'te_coverage':coverage_te,'te_R2':metrics_te[0],'te_MAE':metrics_te[1],
                   'te_RMSE':metrics_te[2],}
    return result_dict



