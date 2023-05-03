# qsar
#the folloewed is required 

python 3.7.4

rdkit 2020.09.1

scikit-learn 1.0.2

torch 1.10.0


#preprepare dataset

python data_preprocess.py --task sar --data_path './dataset1.csv'

#build models

python modelling.py --Task sar --trOte_path './sar_trOte_ran.csv' --data_path './dataset1_1.csv' --ecfp True

#cluster

python cluster.py --path './dataset1_1.csv'
