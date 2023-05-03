import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import  AllChem,MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def smiles_to_mol(path):
    data = pd.read_csv(path)
    smiles = data['smiles']
    mols = []
    for i in smiles:
        mole = Chem.MolFromSmiles(i)
        if mole is None:
            print(i)
        else:
            mols.append(mole)
    print("从文件中读取了 {:d} 个分子".format(len(mols)))
    return mols

def cal_maccs(mols):
    MACCSFps = [list(MACCSkeys.GenMACCSKeys(m).ToBitString()[1:]) for m in mols]
    columns = ["maccs" + str(i) for i in range(1,167)]
    maccs = pd.DataFrame(MACCSFps,columns=columns,dtype=np.int8)
    return maccs

def cal_ecfp4(mols):
    fps = [list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024,useChirality=True).ToBitString()) for m in mols]
    columns = ["ecfp_" + str(i) for i in range(1,len(fps[0])+1)]
    ecfp4 = pd.DataFrame(fps,columns=columns,dtype=np.int8)
    return ecfp4

def cal_2Drdkit(mols):
    rdkit_2d_desc = []
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    header = calc.GetDescriptorNames()
    for i in range(len(mols)):
        ds = calc.CalcDescriptors(mols[i])
        rdkit_2d_desc.append(ds)
    df = pd.DataFrame(rdkit_2d_desc,columns=header)
    return df

