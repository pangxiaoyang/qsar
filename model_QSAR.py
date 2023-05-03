
from math import sqrt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,make_scorer
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import warnings,re

def show_metrics(y_true, y_pred):
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    metrics = [R2,MAE,RMSE]
    return metrics

def evaluation(clf,tr_x,tr_y,te_x,te_y):
    tr_pre_y = clf.predict(tr_x)
    te_pre_y = clf.predict(te_x)

    metrics_tr = show_metrics(tr_y.tolist(), tr_pre_y.tolist())
    metrics_te = show_metrics(te_y.tolist(), te_pre_y.tolist())

    result_info = {'tr_R2':metrics_tr[0],'tr_MAE':metrics_tr[1],
                   'tr_RMSE':metrics_tr[2],'te_R2':metrics_te[0],'te_MAE':metrics_te[1],
                   'te_RMSE':metrics_te[2],}
    return tr_pre_y,te_pre_y,result_info
    
    
def MLR(tr_x,tr_y,te_x,te_y,model_path):
    model = LinearRegression()
    model.fit(tr_x, tr_y)
    result_dict = evaluation(model,tr_x,tr_y,te_x,te_y)
    return result_dict


def RF_R(tr_x,tr_y,te_x,te_y):
    warnings.filterwarnings("ignore")
    n_estimators = []
    for i in range(10,101):
        n_estimators.append(i)

    criterion = ['mse','mae']
    max_features = ["auto", "sqrt", "log2"]
    max_leaf_nodes = []
    for i in range(50,151,10):
        max_leaf_nodes.append(i)

    grid_dict = {'n_estimators':n_estimators,'criterion':criterion,'max_features':max_features,
                 'max_leaf_nodes':max_leaf_nodes}
    grid_score_r2 = make_scorer(r2_score,greater_is_better=True)
    grid_cv = KFold(n_splits=5, shuffle=True, random_state=0)

    grid_rfc = RandomForestRegressor()
    Grid = GridSearchCV(grid_rfc, grid_dict, scoring=grid_score_r2, cv=grid_cv)
    Grid.fit(tr_x, tr_y)
    result = evaluation(Grid.best_estimator_,tr_x,tr_y,te_x,te_y)
    best_params = Grid.best_params_
    best_params['descriptors_num']=len(tr_x.columns)
    tr_pre_y,te_pre_y,result_dict = dict(best_params, **result)
    return tr_pre_y,te_pre_y,result_dict,Grid.best_estimator_

def SVM_R(tr_x,tr_y,te_x,te_y):
    warnings.filterwarnings("ignore")
    C = []
    for i in range(-10,11,2):
        C.append(2**i)

    epsilon = []
    for i in range(-30,0,2):
        epsilon.append(2**i)

    grid_dict = {'C':C, 'gamma':C, 'epsilon':epsilon}
    grid_score_r2 = make_scorer(r2_score,greater_is_better=True)
    grid_cv = KFold(n_splits=5, shuffle=True, random_state=0)

    grid_svr = SVR()
    Grid = GridSearchCV(grid_svr, grid_dict, scoring=grid_score_r2, cv=grid_cv)
    Grid.fit(tr_x, tr_y)
    result = evaluation(Grid.best_estimator_,tr_x,tr_y,te_x,te_y)
    best_params = Grid.best_params_
    best_params['descriptors_num']=len(tr_x.columns)
    tr_pre_y,te_pre_y,result_dict = dict(best_params, **result)
    return tr_pre_y,te_pre_y,result_dict,Grid.best_estimator_

def RFR_training(tr_x,tr_y,te_x,te_y):
    max_R2 = 0.5
    for i in range(10):
        tr_pre_y,te_pre_y,result_dict,model = RF_R(tr_x,tr_y,te_x,te_y)
        if result_dict['te_R2'] >max_R2:
            best_result = result_dict
            tr_pred = tr_pre_y
            te_pred = te_pre_y
    return best_result,tr_pred,te_pred

def SVR_training(tr_x,tr_y,te_x,te_y,):
    max_R2 = 0.5
    for i in range(10):
        tr_pre_y,te_pre_y,result_dict,model = SVM_R(tr_x,tr_y,te_x,te_y)
        if result_dict['te_R2'] >max_R2:
            best_result = result_dict
            tr_pred = tr_pre_y
            te_pred = te_pre_y
    return best_result,tr_pred,te_pred


def prepare(tr_x,tr_y):
    x_arr = tr_x.values
    y_arr = tr_y.values
    x = torch.tensor(x_arr).float()
    y = torch.tensor(y_arr).float()
    dataset = TensorDataset(x,y)
    loader = DataLoader(dataset=dataset,batch_size=25,num_workers=0)
    return loader


def fit(model,loss_func,optimizer,tr_loader,val_loader,epochs):
    val_r2 = []
    for epoch in range(epochs):
        val_ypred = []
        val_y = []
        for i,data in enumerate(tr_loader,0):
            model.zero_grad()
            inputs,labels = data
            y_pred_tr = model(inputs)
            loss = loss_func(y_pred_tr,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            model.eval()
            for j,data in enumerate(val_loader,0):
                inputs,labels = data
                y_pred_val = model(inputs)
                y_pred_val = y_pred_val.data.numpy().tolist()
                val_y.extend(labels)
                val_ypred.extend(y_pred_val)
            r2_val = r2_score(val_y, val_ypred)
        val_r2.append(r2_val)
        max_r2_val = max(val_r2)
        a = int(val_r2.index(max_r2_val))
        if epoch-a == 30:
            break
    return model

###用于输出最优模型指标        
def model_pred(best_model,te_loader):
    with torch.no_grad():
        te_labels = []
        te_pred = []
        for i,data in enumerate(te_loader,0):
            inputs,labels = data
            y_pred = best_model(inputs)
            y_pred = y_pred.numpy().tolist()
           
            te_pred.extend(y_pred)
            te_labels.extend(labels)
        metrics = show_metrics(te_labels, te_pred)
    return te_pred,metrics


class MyNet(nn.Module):
    
    def __init__(self,inputs,l):
        super(MyNet,self).__init__()
        self.fc_1 = nn.Linear(inputs,l[0])
        self.bn_1 = nn.BatchNorm1d(l[0])
        self.fc_2 = nn.Linear(l[0],l[1])
        self.bn_2 = nn.BatchNorm1d(l[1])
        self.fc_3 = nn.Linear(l[1],l[2])
        self.bn_3 = nn.BatchNorm1d(l[2])
        self.fc_4 = nn.Linear(l[2],1)
        
    def forward(self,input_data):
        x = F.relu(self.bn_1(self.fc_1(input_data)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = F.relu(self.bn_3(self.fc_3(x)))
        x = self.fc_4(x)
        return x
    
    
def DNNR_training(tr_x,tr_y,te_x,te_y,l):
    train_loader = prepare(tr_x, tr_y)
    test_loader = prepare(te_x, te_y)
    
    model = MyNet(len(tr_x.columns),l)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.01)
    loss_func = torch.nn.MSELoss()
    max_R2 = 0.5
    for i in range(10):
        for train_index, val_index in KFold(5, shuffle=False).split(tr_x):
            train_x, val_x = tr_x.iloc[train_index], tr_x.iloc[val_index]
            train_y, val_y = tr_y.iloc[train_index], tr_y.iloc[val_index]
            tr_loader = prepare(train_x, train_y)
            val_loader = prepare(val_x, val_y)
            
            best_model = fit(model,loss_func,optimizer,tr_loader,val_loader,epochs = 1000)
        
        te_metrics = model_pred(best_model,test_loader)
        if te_metrics[0] > max_R2:
            max_R2 = te_metrics[0]
            tr_pred,metrics_tr = model_pred(best_model,train_loader)
            te_pred,metrics_te = model_pred(best_model,test_loader)
            
            result_dict = {'tr_R2':metrics_tr[0],'tr_MAE':metrics_tr[1],
                    'tr_RMSE':metrics_tr[2],'te_R2':metrics_te[0],'te_MAE':metrics_te[1],
                    'te_RMSE':metrics_te[2],}
    return result_dict,tr_pred,te_pred

    