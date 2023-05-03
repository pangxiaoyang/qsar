"""
Usage:
    DeepCoy.py [options]

Options:
    -h --help                Show this screen
    --Task NAME          Task name: sar, qsar
    --data_path FILE       the file of 3018 inhibitors which include smiles and activity
    --trOte_path          the file of trOte
    --maccs             True or False
    --ecfp             True or False
    --rdkit             True or False
    --des_path           the file of calculated descriptors
    --tr_prob_path          the file of tr_proba_all
    --tr_prob_path          the file of te_proba_all
"""

import pandas as pd
import numpy as np
from QSAR_package.data_scale import dataScale
from model_classification import RFC_training,SVC_training,DNNC_training
from descriptors_calculation import des_screen
from QSAR_package.data_split import extractData
from sklearn.preprocessing import MinMaxScaler
from model_QSAR import RFR_training,SVR_training,DNNR_training
from docopt import docopt
from descriptors_calculation import cal_2Drdkit,smiles_to_mol,cal_maccs,cal_ecfp4
from dstd_pro import cal_AD

def extract(tr_x,te_x,des_list):
    tr_extract_x = tr_x.loc[:,des_list]
    te_extract_x = te_x.loc[:,des_list]
        
    return tr_extract_x, te_extract_x 

def scale(tr_x,te_x):
    if (tr_x.min() == 0 and tr_x.max() == 1):
        tr_scaled_x = tr_x
        te_scaled_x = te_x
    else:
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        tr_scaled_x = scaler.fit_transform(tr_x)
        te_scaled_x = scaler.transform(te_x)
    return tr_scaled_x,te_scaled_x


def SAR(data,activity,trOte_path,l):
    trOte = pd.read_csv(trOte_path)
    tr_x = data[np.where(trOte['trOte']=='tr')[0].tolist()]
    tr_y = activity[np.where(trOte['trOte']=='tr')[0].tolist()]
    te_x = data[np.where(trOte['trOte']=='te')[0].tolist()]
    te_y = activity[np.where(trOte['trOte']=='te')[0].tolist()]
    des_list = des_screen(tr_x,tr_y)
    tr_x,te_x = scale(extract(tr_x,te_x,des_list))
    tr_proba_all = []
    te_proba_all = []
    result = []
    best_result,tr_proba,te_proba = RFC_training(tr_x,tr_y,te_x,te_y)
    result.append(best_result)
    tr_proba_all.qppend(tr_proba)
    te_proba_all.qppend(te_proba)
    print('RF model complete')
    best_result,tr_proba,te_proba = SVC_training(tr_x,tr_y,te_x,te_y)
    result.append(best_result)
    tr_proba_all.qppend(tr_proba)
    te_proba_all.qppend(te_proba)
    print('SVM model complete')
    best_result,tr_proba,te_proba = DNNC_training(tr_x,tr_y,te_x,te_y,l)
    result.append(best_result)
    tr_proba_all.qppend(tr_proba)
    te_proba_all.qppend(te_proba)
    print('DNN model complete')
    return result,tr_proba_all,te_proba_all

def QSAR(data,pIC50,trOte_path,l):
    trOte = pd.read_csv(trOte_path)
    tr_x = data[np.where(trOte['trOte']=='tr')[0].tolist()]
    tr_y = pIC50[np.where(trOte['trOte']=='tr')[0].tolist()]
    te_x = data[np.where(trOte['trOte']=='te')[0].tolist()]
    te_y = pIC50[np.where(trOte['trOte']=='te')[0].tolist()]
    des_list = des_screen(tr_x,tr_y)
    tr_x,te_x = scale(extract(tr_x,te_x,des_list))
    tr_pred_all = []
    te_pred_all = []
    result = []
    best_result,tr_pred,te_pred = RFR_training(tr_x,tr_y,te_x,te_y)
    result.append(best_result)
    tr_pred_all.qppend(tr_pred)
    te_pred_all.qppend(te_pred)
    print('RF model complete')
    best_result,tr_pred,te_pred = SVC_training(tr_x,tr_y,te_x,te_y)
    result.append(best_result)
    tr_pred_all.qppend(tr_pred)
    te_pred_all.qppend(te_pred)
    print('SVM model complete')
    best_result,tr_pred,te_pred = DNNC_training(tr_x,tr_y,te_x,te_y,l)
    result.append(best_result)
    tr_pred_all.qppend(tr_pred)
    te_pred_all.qppend(te_pred)
    print('DNN model complete')
    return result,tr_pred_all,te_pred_all


if __name__ == "__main__":
    args = docopt(__doc__)
    Task=args.get('--Task')
    data_path=args.get('--data_path')
    trOte_path=args.get('--trOte_path')
    try:
        des_path=args.get('--des_path')
        tr_prob_path=args.get('--tr_prob_path')
        te_prob_path=args.get('--te_prob_path')
        maccs=args.get('--maccs')
        ecfp=args.get('--ecfp')
        rdkit=args.get('--ecfp')

    except:
        des_path=False
        tr_prob_path=False
        te_prob_path=False
        maccs=False
        ecfp=False
        rdkit=False
        
    if Task == 'sar':
        activity = pd.read_csv(data_path)['activity']
        result_all = []
        if des_path:
            data = pd.read_csv(des_path)
            l = [100,50,20]
            result,tr_proba_all,te_proba_all = SAR(data,activity,trOte_path,l)
            result_all.extend(result)
        if maccs:
            maccs = cal_maccs(smiles_to_mol(data_path))
            l = [200,100,50]
            result,tr_proba_all,te_proba_all = SAR(maccs,activity,trOte_path,l)
            result_all.extend(result)
        if ecfp:
            ecfp = cal_ecfp4(smiles_to_mol(data_path))
            l = [500,200,50]
            result,tr_proba_all,te_proba_all = SAR(ecfp,activity,trOte_path,l)
            result_all.extend(result)
        if rdkit:
            rdkit = cal_2Drdkit(smiles_to_mol(data_path))
            l = [100,50,20]
            result,tr_proba_all,te_proba_all = SAR(rdkit,activity,trOte_path,l)
            result_all.extend(result)
        print(pd.DataFrame(result_all))
    
    else:
        pIC50 = pd.read_csv(data_path)['pIC50']
        result_all = []
        l = [100,50,20]
        if des_path:
            data = pd.read_csv(des_path)
            result,tr_pred_all,te_pred_all = SAR(data,activity,trOte_path,l)
            result_all.extend(result)
        if rdkit:
            rdkit = cal_2Drdkit(smiles_to_mol(data_path))
            result,tr_pred_all,te_pred_all = SAR(rdkit,activity,trOte_path,l)
            result_all.extend(result)
        print(pd.DataFrame(result_all))

        
    if tr_prob_path:
        tr_proba = pd.read_csv(tr_prob_path)
        te_proba = pd.read_csv(te_prob_path)
        activity = pd.read_csv(data_path)['activity']
        trOte = pd.read_csv(trOte_path)
        tr_y = activity[np.where(trOte['trOte']=='tr')[0].tolist()]
        te_y = activity[np.where(trOte['trOte']=='te')[0].tolist()]
        result = cal_AD(tr_proba,tr_y,te_proba,te_y)
        print(result)
            



    
    






