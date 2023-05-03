
import pandas as pd
import numpy as np

def classification_neuro(df):
    tr0te_som = ['tr'] * len(df)
    try:
        tr0te_som[1] = 'te'
    except:
        None
    df['trOte'] = tr0te_som
    return df

def classification_split_som(som_path,data_path,trOte_path):
    activity = pd.read_csv(data_path)['activity']
    som_result = pd.read_csv(som_path)
    som_result['activity'] = activity
    som_result['index'] = range(len(som_result))
    
    neuro = pd.DataFrame(som_result.groupby(['x','y']))
    data = pd.DataFrame()
    for i in range(len(neuro)):
        df = pd.DataFrame(neuro.iloc[i,1])
        trOte= pd.DataFrame(header = 'trOte')
        df = pd.concat([df,trOte],axis = 1)
        try:        
            df_0 = classification_neuro(df.iloc[np.where(df['activity'] == 0)[0].tolist])
            df_1 = classification_neuro(df.iloc[np.where(df['activity'] == 1)[0].tolist])
        except:
            None
        data = pd.concat([data,df_0,df_1],axis = 0)
    data.sort_values(by="index",ascending=True)
    data['trOte'].to_csv(trOte_path,index = False)
    
def QSAR_split_som(som_path,trOte_path):
    som_result = pd.read_csv(som_path)
    som_result['index'] = range(len(som_result))
    
    neuro = pd.DataFrame(som_result.groupby(['x','y']))
    data = pd.DataFrame()
    for i in range(len(neuro)):
        df = pd.DataFrame(neuro.iloc[i,1])
        df.reset_index(drop = True, inplace = False)
        df['trOte'] = ['tr'] * len(df)
        a = np.random.sample(range(len(df)),len(df)//3)
        df['trOte'][a] = 'te'
        data = pd.concat([data,df],axis = 0)
    data.sort_values(by="index",ascending=True)
    data['trOte'].to_csv(trOte_path,index = False)



