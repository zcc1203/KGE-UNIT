################################
#This script provide a demo of MPNN_CNN & DeepDTI, the runtime on one fold mainly takes 3~5 hours (V100). 

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import metrics
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.decomposition import PCA
#Load Data
################################################################
# dt_08 = pd.read_csv('./data/yamanishi_08/dt_all_08.txt',delimiter='\t',header=None)
# dt_08.columns = ['head','relation','tail']

# df_drug = pd.read_csv('./data/BioKG/comp_struc.csv')
# df_proseq = pd.read_csv('./data/BioKG/pro_seq.csv')
# df_proseq.columns = ['pro_id','seq']

# pro_id = df_proseq['pro_id']

df_drug = pd.read_csv('./data/luo/feature/drug_smiles.csv')

df_drug_smiles = df_drug['smiles']


fp_id = pd.read_csv('./data/luo/feature/drug_smiles.csv')['id']
fp_id_list = []
fpid_path = './data/luo/mapping/drug.txt'
fp_drugid = open(fpid_path,'r')
for idx,line in enumerate(fp_drugid.readlines()):
    fp_id_list.append("Drug::"+str(idx))
fp_id = pd.DataFrame(fp_id_list,columns = ['drug_id'])
df_drug = pd.concat([fp_id,df_drug_smiles],axis=1)

pro_id_list = []
proid_path = './data/luo/mapping/protein.txt'
fp_proid = open(proid_path,'r')
for idx,line in enumerate(fp_proid.readlines()):
    pro_id_list.append("Protein::"+str(idx))
pro_id = pd.DataFrame(pro_id_list,columns = ['pro_id'])

#pro_feats = np.loadtxt('./data/luo/feature/seq.txt')
fp = open('./data/luo/feature/seq.txt','r')
pro_feats = []
lines = fp.readlines()
for line in lines:
    line = line.strip()
    pro_feats.append(line)

pro_feats = pd.DataFrame(pro_feats,columns = ['seq'])
df_proseq = pd.concat([pro_id,pro_feats],axis=1)

#define function
################################

def get_struc(data,df_drug,df_proseq):
    drug_struc = pd.merge(data,df_drug,how='left',left_on='head',right_on='drug_id')['smiles'].values
    pro_struc = pd.merge(data,df_proseq,how='left',left_on='tail',right_on='pro_id')['seq'].values
    #return data['label'].values,drug_struc,pro_struc
    return drug_struc,pro_struc


def roc_auc(y,pred):
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, _ = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


data_path = './data/luo/data_folds/cold_start_1_1/'  
def load_data(i):
    train = pd.read_csv(data_path+'train_fold_'+str(i+1)+'.csv')
    test = pd.read_csv(data_path+'test_fold_'+str(i+1)+'.csv')
    return train,test

#need to be adjusted when changing methods
#drug_encoding, target_encoding = 'MPNN', 'CNN'
drug_encoding, target_encoding = 'CNN', 'CNN'
def get_input(train_all,test_all):
    train_label = train_all['label']
    test_label = test_all['label']
    train_re, valid_re, y_train, y_valid = train_test_split(train_all[['head','relation','tail']],train_label,test_size=0.01, 
                                                                random_state=0,
                                                                stratify=train_label)
    train_drug_feats,train_pro_feats = get_struc(train_re,df_drug,df_proseq)
    valid_drug_feats,valid_pro_feats = get_struc(valid_re,df_drug,df_proseq)
    test_drug_feats,test_pro_feats = get_struc(test_all,df_drug,df_proseq)
    train = utils.data_process(train_drug_feats, train_pro_feats, y_train, 
                                drug_encoding, target_encoding, 
                                split_method='no_split',
                                random_seed = 0)
    valid = utils.data_process(valid_drug_feats, valid_pro_feats, y_valid, 
                            drug_encoding, target_encoding, 
                            split_method='no_split',
                            random_seed = 0)                            
    test = utils.data_process(test_drug_feats, test_pro_feats, test_label, 
                                drug_encoding, target_encoding, 
                                split_method='no_split',
                                random_seed = 0)
    return train,valid,test


######################################## Training
'''
#parameters for MPNN_CNN
config = utils.generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                        cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 1, 
                         LR = 0.001, 
                         batch_size = 2500,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3, 
                         cnn_target_filters = [32,64,64],
                         cnn_target_kernels = [4,8,8]
                        )
'''

#parameters for DeepDTI
config = utils.generate_config(drug_encoding, target_encoding, 
                            cls_hidden_dims = [1024,1024,512],
                            train_epoch = 100, 
                            LR = 0.001, 
                            batch_size = 1000, 
                            cnn_drug_filters = [32,64,96],
                            cnn_drug_kernels = [4,8,12], 
                            cnn_target_filters = [32,64,96], 
                            cnn_target_kernels = [4,8,12])


for i in range(10):
    print(i)
    train,test = load_data(i)
    train_input,valid_input,test_input = get_input(train,test)
    model = models.model_initialize(**config)
    model.train(train_input,valid_input)
