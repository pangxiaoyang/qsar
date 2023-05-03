"""
Usage:
    DeepCoy.py [options]
    
Options:
    -h --help                Show this screen.
    --task str                  task(sar, qsar)
    --data_path file         Path to data file
    --som_path file          Path to som_result file
"""


import pandas as pd
from docopt import docopt
import math
from QSAR_package.data_split import randomSpliter
from split_trOte_som import classification_split_som,QSAR_split_som



def cal_pIC50(path):
    data = pd.read_csv(path)
    IC50 = data['IC50']
    pIC50 = []
    for i in range(0,len(IC50)):
        pIC50.append((9-math.log(IC50.tolist()[i])),10)
    data['pIC50'] = pIC50
    data.to_csv(path[:-4]+'_1.csv',index = False)

def cal_activity(path,threshold):
    data = pd.read_csv(path)
    IC50 = data['IC50']
    activity = []
    for i in range(0,len(IC50)):
        if IC50.iloc[i,0] <= threshold:
            activity.append(1)
        elif IC50.iloc[i,0] > threshold:
            activity.append(0)
        else:
            activity.append(-1)
    data['activity'] = activity
    data.to_csv(path[:-4]+'_1.csv',index = False)

def trOte_ran(path,trOte_path,label):
    spliter = randomSpliter(test_size=0.25,random_state=0)
    spliter.ExtractTotalData(path,label_name=label) #注意指定标签（活性）列的列名
    spliter.saveTrainTestLabel(trOte_path) 


if __name__ == "__main__":
    # Parse args
    args = docopt(__doc__)
    path = args.get('--data_path') 
    som_path = args.get('--som_path')
    task = args.get('--task')
    trOte_path = './'+task+'_trOte_ran.csv'
    threshold = 200
    
    if task == 'sar':
        cal_activity(path,threshold)
        trOte_ran(path,trOte_path,'activity')
        try:
            classification_split_som(som_path,path,trOte_path)
        except:
            None
    elif task == 'qsar':
        cal_pIC50(path)
        trOte_ran(path,trOte_path,'pIC50')
        try:
            QSAR_split_som(som_path,trOte_path)
        except:
            None

