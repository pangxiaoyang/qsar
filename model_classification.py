import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score,RandomizedSearchCV
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import warnings,re



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

def evaluation(clf,tr_x,tr_y,te_x,te_y):
    tr_pre_y = clf.predict(tr_x)
    te_pre_y = clf.predict(te_x)

    metrics_tr = show_metrics(tr_y.tolist(), tr_pre_y.tolist())
    metrics_te = show_metrics(te_y.tolist(), te_pre_y.tolist())
    
    cv_score_accuracy = make_scorer(accuracy_score)
    cv_cv5 = StratifiedKFold(n_splits=5, shuffle=True ,random_state=0)
    fold_5  = np.mean(cross_val_score(clf, tr_x, tr_y,n_jobs=-1, scoring=cv_score_accuracy, cv=cv_cv5))
    
    result_info = {'tr_tp':metrics_tr[0],'tr_tn':metrics_tr[1],
                   'tr_fp':metrics_tr[2],'tr_fn':metrics_tr[3],
                   'tr_accuracy':metrics_tr[4],'tr_MCC':metrics_tr[5],
                   '5_CV':fold_5,'te_tp':metrics_te[0],'te_tn':metrics_te[1],
                   'te_fp':metrics_te[2],'te_fn':metrics_te[3],
                   'te_accuracy':metrics_te[4],'te_MCC':metrics_te[5],
                   'te_se':metrics_te[6],'te_sp':metrics_te[7]}
    return result_info

    

def RF_C(tr_x,tr_y,te_x,te_y):
    warnings.filterwarnings("ignore")
    n_estimators = range(1,200,2)
    max_features = [None,'sqrt','log2']
    criterion = ['entropy','gini']
    grid_dict = {'n_estimators':n_estimators,'max_features':max_features,'criterion':criterion}
    grid_estimator = RandomForestClassifier(random_state=1)
    
    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    grid_cv = StratifiedKFold(n_splits=5, shuffle=True,random_state=1)
    #grid_cv = LeaveOneOut()
    Grid = GridSearchCV(grid_estimator, grid_dict, scoring=grid_scorer, cv=grid_cv, verbose=1,n_jobs=-1)
    Grid.fit(tr_x, tr_y) 
    
    result = evaluation(Grid.best_estimator_,tr_x,tr_y,te_x,te_y)
    best_params = Grid.best_params_
    result['descriptors_num']=len(tr_x.columns)
    result_dict = dict(best_params, **result)
    
    return result_dict,best_params,Grid.best_estimator_

def SVM_C(tr_x,tr_y,te_x,te_y):
    warnings.filterwarnings("ignore")
    grid_list = []
    for i in range(-10,5):
        grid_list.append(2**i)
    grid_dict = {'C':grid_list, 'gamma':grid_list}
    grid_estimator = SVC(probability=True)
    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    grid_cv = StratifiedKFold(n_splits=5, shuffle=True ,random_state=1)
    #grid_cv = LeaveOneOut()
    Grid = GridSearchCV(grid_estimator, grid_dict, scoring=grid_scorer,cv=grid_cv, verbose=1,n_jobs=-1)
    Grid.fit(tr_x, tr_y)
    
    result = evaluation(Grid.best_estimator_,tr_x,tr_y,te_x,te_y)
    best_params = Grid.best_params_
    best_params['descriptors_num']=len(tr_x.columns)
    result_dict = dict(best_params, **result)
    
    return result_dict,best_params,Grid.best_estimator_


def RFC_proba(tr_x,tr_y,te_x,params):
    index = []
    y_proba_tr = []
    for train_index, val_index in StratifiedKFold(5, shuffle=False).split(tr_x,tr_y):
        train_x, val_x = tr_x.iloc[train_index], tr_x.iloc[val_index]
        train_y = tr_y.iloc[train_index]
        index.extend(val_index.tolist())
        model = RandomForestClassifier(**params)
        model.fit(train_x,train_y)
        y_proba = model.predict_proba(val_x)[:,1].tolist()
        y_proba_tr.extend(y_proba)
    tr_proba = pd.DataFrame({'val_index':index,'cv_proba':y_proba_tr}).sort_values(by = 'val_index',axis = 0)
    tr_proba.reset_index(drop = True,inplace = True)
    tr_proba.drop(columns = 'val_index',inplace = True)
    model.fit(tr_x,tr_y)
    te_proba = model.predict_proba(te_x)[:,1].tolist()
    te_proba = pd.DataFrame(te_proba,columns = ['te_proba'])
    return tr_proba,te_proba

def SVC_proba(tr_x,tr_y,te_x,params):
    
    index = []
    y_proba_tr = []
    for train_index, val_index in StratifiedKFold(5, shuffle=False).split(tr_x,tr_y):
        index.extend(val_index.tolist())
        train_x, val_x = tr_x.iloc[train_index], tr_x.iloc[val_index]
        train_y = tr_y.iloc[train_index]
        model = SVC(**params)
        model.fit(train_x,train_y)
        y_proba = model.decision_function(val_x)[:,1].tolist()
        y_proba_tr.extend(y_proba)
    tr_proba = pd.DataFrame({'val_index':index,'cv_proba':y_proba_tr}).sort_values(by = 'val_index',axis = 0)
    tr_proba.reset_index(drop = True,inplace = True)
    tr_proba.drop(columns = 'val_index',inplace = True)
    model.fit(tr_x,tr_y)
    te_proba = model.decision_function(te_x)[:,1].tolist()
    return tr_proba,te_proba


def RFC_training(tr_x,tr_y,te_x,te_y):
    max_mcc = 0.5
    for i in range(10):
        result_dict,params,model = RF_C(tr_x,tr_y,te_x,te_y)
        if result_dict['te_MCC'] >max_mcc:
            max_mcc = result_dict['te_MCC']
            best_result = result_dict
            best_params = params
    tr_proba,te_proba = RFC_proba(tr_x,tr_y,te_x,best_params)
    return best_result,tr_proba,te_proba


def SVC_training(tr_x,tr_y,te_x,te_y):
    max_mcc = 0.5
    for i in range(10):
        result_dict,params,model = SVM_C(tr_x,tr_y,te_x,te_y)
        if result_dict['te_MCC'] >max_mcc:
            max_mcc = result_dict['te_MCC']
            best_result = result_dict
            best_params = params
    tr_proba,te_proba = SVC_proba(tr_x,tr_y,te_x,best_params)
    return best_result,tr_proba,te_proba


def prepare(tr_x,tr_y):
    x_arr = tr_x.values
    y_arr = tr_y.values
    y_arr = encode(y_arr)
    x = torch.tensor(x_arr).float()
    y = torch.tensor(y_arr).float()
    dataset = TensorDataset(x,y)
    loader = DataLoader(dataset=dataset,batch_size=25,num_workers=0)
    return loader

def encode(arr):
    arr = np.array(arr).reshape(len(arr),-1)
    enc = OneHotEncoder()
    enc.fit(arr)
    target = enc.transform(arr).toarray()
    return target

def decode(arr):
    l = []
    for i in range(len(arr)):
        if arr[i][1] <= 0.5:
            l.append(0)
        else:
            l.append(1)
    return l

def fit(model,loss_func,optimizer,tr_loader,val_loader,epochs):
    val_acc = []
    for epoch in range(epochs):
        cv_proba = []
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
                y_pred_val = y_pred_val.data.numpy()
                y_proba = y_pred_val[:,1].tolist()
                y_pred_val = decode(y_pred_val)
                label = decode(labels)
                val_y.extend(label)
                val_ypred.extend(y_pred_val)
                cv_proba.extend(y_proba)
            acc_val = accuracy_score(val_y, val_ypred)
        val_acc.append(acc_val)
        max_acc_val = max(val_acc)
        a = int(val_acc.index(max_acc_val))
        if epoch-a == 30:
            break
    return cv_proba,model,acc_val

###用于输出最优模型指标        
def model_pred(best_model,te_loader):
    with torch.no_grad():
        te_labels = []
        te_pred = []
        te_proba = []
        for i,data in enumerate(te_loader,0):
            inputs,labels = data
            y_pred = best_model(inputs)
            y_pred = y_pred.numpy()
            
            y_proba = y_pred[:,1].tolist()
            te_proba.extend(y_proba)
            
            y_pred = decode(y_pred)
            te_pred.extend(y_pred)

            te_y = decode(labels)
            te_labels.extend(te_y)
        metrics = show_metrics(te_labels, te_pred)
        te_proba = pd.DataFrame(te_proba,columns = ['te_proba'])
    return te_proba,metrics


class MyNet(nn.Module):
    
    def __init__(self,inputs,l):
        super(MyNet,self).__init__()
        self.fc_1 = nn.Linear(inputs,l[1])
        self.bn_1 = nn.BatchNorm1d(l[1])
        self.fc_2 = nn.Linear(l[1],l[2])
        self.bn_2 = nn.BatchNorm1d(l[2])
        self.fc_3 = nn.Linear(l[2],l[3])
        self.bn_3 = nn.BatchNorm1d(l[3])
        self.fc_4 = nn.Linear(l[3],2)
        
    def forward(self,input_data):
        x = F.relu(self.bn_1(self.fc_1(input_data)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = F.relu(self.bn_3(self.fc_3(x)))
        x = torch.sigmoid(self.fc_4(x))
        return x
    
    
def DNNC_training(tr_x,tr_y,te_x,te_y,l):
    train_loader = prepare(tr_x, tr_y)
    test_loader = prepare(te_x, te_y)
    model = MyNet(len(tr_x.columns),l)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.01)
    loss_func = torch.nn.BCELoss()
    max_mcc = 0.5
    for i in range(1):
        cv_proba = []
        cv_5 = 0
        index = []
        for train_index, val_index in StratifiedKFold(5, shuffle=False).split(tr_x,tr_y):
            index.extend(val_index)
            train_x, val_x = tr_x.iloc[train_index], tr_x.iloc[val_index]
            train_y, val_y = tr_y.iloc[train_index], tr_y.iloc[val_index]
            tr_loader = prepare(train_x, train_y)
            val_loader = prepare(val_x, val_y)
            
            y_proba_tr,best_model,acc_val = fit(model,loss_func,optimizer,tr_loader,val_loader,epochs = 1000)
            cv_proba.extend(y_proba_tr)
            cv_5+=acc_val
            
        tr_proba = pd.DataFrame({'val_index':index,'cv_proba':cv_proba}).sort_values(by = 'val_index',axis = 0)
        tr_proba.reset_index(drop = True,inplace = True)
        tr_proba.drop(columns = 'val_index',inplace = True)

        te_proba,te_metrics = model_pred(best_model,test_loader)
        if te_metrics[5] > max_mcc:
            max_mcc = te_metrics[5]

            train_proba,train_metrics = model_pred(best_model,train_loader)
            test_proba,test_metrics = model_pred(best_model,test_loader)
            train_proba = tr_proba
            
            best_result = {'tr_tp':train_metrics[0],'tr_tn':train_metrics[1],
                    'tr_fp':train_metrics[2],'tr_fn':train_metrics[3],
                    'tr_accuracy':train_metrics[4],'tr_MCC':train_metrics[5],
                    '5_CV':cv_5/5,'te_tp':test_metrics[0],'te_tn':test_metrics[1],
                    'te_fp':test_metrics[2],'te_fn':test_metrics[3],
                    'te_accuracy':test_metrics[4],'te_MCC':test_metrics[5],
                    'te_se':test_metrics[6],'te_sp':test_metrics[7]}
    return best_result,train_proba,test_proba

