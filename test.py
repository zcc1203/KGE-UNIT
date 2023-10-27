import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils import data
from dataset import Data_Encoder
from config import KGE_UNIT_config
from argparse import ArgumentParser
from models import MultiTaskLoss
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline 
from pykeen.datasets.nations import NATIONS_TRAIN_PATH,NATIONS_TEST_PATH
import pathlib
HERE = pathlib.Path(__file__).resolve().parent
print(HERE)
print(NATIONS_TRAIN_PATH)
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,precision_recall_curve,auc

KGE_NAME = 'conve'
DATASET_PATH = './data/luo/'
SAVE_PATH = './run1/'
parser = ArgumentParser(description='KEG_UNIT Testing.')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedd_dim', type=int,
                        default=128, help='the dim of embedding')
parser.add_argument('--gpu',type=int,default=1,help = 'the number of GPUS')
parser.add_argument('--scenario',type=str,default ='warm',help='warm/cold')
parser.add_argument('--kge',type=str,help='kge model path')
parser.add_argument('--kgeunit',type=str,help='kge-unit model path')

config = KGE_UNIT_config()
args = parser.parse_args()
config['batch_size'] = args.batch_size
config['embedd_dim'] = args.embedd_dim
print("123123123123123\n")
######output path 

best_model_path_kge = os.path.join(SAVE_PATH,'KGE/')
best_model_path_kgeunit = os.path.join(SAVE_PATH,'KGE-UNIT/')
if not os.path.exists(best_model_path_kge):
    os.makedirs(best_model_path_kge)
if not os.path.exists(best_model_path_kgeunit):
    os.makedirs(best_model_path_kgeunit)
#####input path
if args.scenario == 'warm':
    data_path =  os.path.join(DATASET_PATH,'data_folds/warm_start_1_1/')
    data_path_ddi = os.path.join(DATASET_PATH,'data_folds/warm_start_1_1_ddi/')
    list2 = [286,318,288,308,308,304,308,294,288,290]
elif args.scenario == 'cold':
    data_path = os.path.join(DATASET_PATH,'data_folds/cold_start_1_1/')
    data_path_ddi = os.path.join(DATASET_PATH,'data_folds/cold_start_1_1_ddi/')
    list2 = [82,96,56,68,108,54,100,95,66,52]

KGE_IDX_PATH = os.path.join(DATASET_PATH,'kg.csv')
kge_path =  os.path.join(DATASET_PATH,'KGE_PATH/')   
kge_cold_path = os.path.join(DATASET_PATH,'KGE_COLD_PATH/')


train_ddi = pd.read_csv(data_path_ddi+'train_fold_'+str(1)+'.csv')[['head','relation','tail','label']]
test_ddi = pd.read_csv(data_path_ddi+'test_fold_'+str(1)+'.csv')[['head','relation','tail','label']]

#all_ddi =train_ddi.append(test_ddi) 
all_ddi = pd.concat([train_ddi,test_ddi])
all_ddi.columns = ['head','relation','tail','label']
######build kg
kg1 = pd.read_csv(os.path.join(DATASET_PATH,'kg_all.txt'),delimiter='\t',header=None)
kg1.columns = ['head','relation','tail']

kg_ddi = kg1[kg1['relation']=='DDI']
dt_08 = kg_dti = kg1[kg1['relation']=='drug_target_interaction']

kg_temp = pd.concat([kg1,dt_08]).drop_duplicates(subset=['head','relation','tail'],keep=False)
kg_temp2 = pd.concat([kg_temp,kg_ddi]).drop_duplicates(subset=['head','relation','tail'],keep=False)
kg = kg_temp2
kg.index = range(len(kg))
kg.columns = ['head','relation','tail']
if args.scenario =='cold':
    kg = pd.read_csv(os.path.join(DATASET_PATH,'kg_cold.csv'),delimiter=',')[['head','relation','tail']]


head_le = LabelEncoder()
tail_le = LabelEncoder()
head_le.fit(dt_08['head'].values)
tail_le.fit(dt_08['tail'].values)
head_le_ddi= LabelEncoder()
tail_le_ddi = LabelEncoder()
head_le_ddi.fit(all_ddi['head'].values)
tail_le_ddi.fit(all_ddi['tail'].values)

mms = MinMaxScaler(feature_range=(0,1))

fp_id = pd.read_csv(os.path.join(DATASET_PATH,'feature/drug_smiles.csv'))['id']
fp_id_list = []
fpid_path = './data/luo/mapping/drug.txt'
fp_drugid = open(fpid_path,'r')
for idx,line in enumerate(fp_drugid.readlines()):
    fp_id_list.append("Drug::"+str(idx))
fp_id = pd.DataFrame(fp_id_list,columns = ['drug_id'])

pro_id_list = []
proid_path = './data/luo/mapping/protein.txt'
fp_proid = open(proid_path,'r')
for idx,line in enumerate(fp_proid.readlines()):
    pro_id_list.append("Protein::"+str(idx))
pro_id = pd.DataFrame(pro_id_list,columns = ['pro_id'])

drug_feats = np.loadtxt('./data/luo/feature/morganfp.txt',delimiter=',')
pro_feats = np.loadtxt('./data/luo/feature/pro_ctd.txt',delimiter=',')


drug_feats = drug_feats[:,:-1].astype(float)
pro_feats = pro_feats[:,:-1].astype(float)
pro_feats_scaled = mms.fit_transform(pro_feats)
pro_feats_scaled2 = PCA(n_components=config['embedd_dim']).fit_transform(pro_feats_scaled)
pro_feats_scaled3 = mms.fit_transform(pro_feats_scaled2)

drug_feats_scaled = mms.fit_transform(drug_feats)
drug_feats_scaled2 = PCA(n_components=config['embedd_dim']).fit_transform(drug_feats_scaled)
drug_feats_scaled3 = mms.fit_transform(drug_feats_scaled2)

fp_df = pd.concat([fp_id,pd.DataFrame(drug_feats_scaled3)],axis=1)
prodes_df = pd.concat([pro_id,pd.DataFrame(pro_feats_scaled3)],axis=1)

def load_data(i):
    train = pd.read_csv(data_path+'train_fold_'+str(i+1)+'.csv')[['head','relation','tail','label']]
    train_pos = train[train['label']==1]
    test = pd.read_csv(data_path+'test_fold_'+str(i+1)+'.csv')[['head','relation','tail','label']]
    test_pos = test[test['label']==1]
    train_ddi = pd.read_csv(data_path_ddi+'train_fold_'+str(i+1)+'.csv')[['head','relation','tail','label']]
    train_ddi_pos = train_ddi[train_ddi['label']==1]
    test_ddi = pd.read_csv(data_path_ddi+'test_fold_'+str(i+1)+'.csv')[['head','relation','tail','label']]
    test_ddi_pos = test_ddi[test_ddi['label']==1]

    data = pd.concat([train_pos,train_ddi_pos,kg])[['head','relation','tail']]
    #data.to_csv(data_path+'train_kg_'+str(i+1)+'.csv',header=None,index=False,sep='\t')
    data1 = pd.concat([test_pos,test_ddi_pos])[['head','relation','tail']]
    #data1.to_csv(data_path+'test_kg_'+str(i+1)+'.csv',header=None,index=False,sep='\t')
    return train,train_pos,test,data,train_ddi,train_ddi_pos,test_ddi

def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

def get_features(data,fp_df,prodes_df,use_pro):
    drug_features = pd.merge(data,fp_df,how='left',left_on='head',right_on='drug_id').iloc[:,4:].values
    # for idx, drug in enumerate(drug_features):
        # for i in range(1029):
            # if drug[i]==1.0:
                # import pdb;pdb.set_trace()
            # else:
                # print(drug[i])
    pro_features = pd.merge(data,prodes_df,how='left',left_on='tail',right_on='pro_id').iloc[:,4:].values
    if use_pro:
        feature = np.concatenate([drug_features,pro_features],axis=1)
    else:
        feature = drug_features
    return drug_features,pro_features

def _eval(y_pred,labels):
    labels=np.array(labels)
    y_pred=np.array(y_pred)
    #y_pred_labels=y_pred.argmax(axis=1)
    y_pred_labels=np.array([0 if i<0.5 else 1 for i in y_pred])
    #y_score=np.array([y_pred[i,index] for i,index in enumerate(y_pred_labels)])
    acc=accuracy_score(labels,y_pred_labels)
    roc_score=roc_auc_score(labels,y_pred)
    pre_score=precision_score(labels,y_pred_labels)
    recall=recall_score(labels,y_pred_labels)
    pr,re,_=precision_recall_curve(labels,y_pred,pos_label=1)
    aupr=auc(re,pr)

    return acc,roc_score,pre_score,recall,aupr
    
    
def _eval_2cls(y_pred,labels):
    pass
def test(data_generator, model,epoch,idx):
    y_pred_dti = []
    y_label_dti = []
    y_pred_ddi = []
    y_label_ddi = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (hf_dti,sf_dti_d,sf_dti_p,hf_ddi,sf_ddi_d1,sf_ddi_d2,label_dti,label_ddi) in enumerate(data_generator):
        dti_pred,ddi_pred =model(hf_dti=hf_dti,
                sf_dti_drug= sf_dti_d,
                sf_dti_protein= sf_dti_p,
                hf_ddi = hf_ddi,
                sf_ddi_d1 = sf_ddi_d1,
                sf_ddi_d2 = sf_ddi_d2,
                dti_labels=label_dti,
                ddi_labels=label_ddi,
                mode='single',
                eval=True)
        #import pdb;pdb.set_trace()
        label_ddi=label_ddi.to('cpu').numpy()
        label_dti = label_dti.to('cpu').numpy()
        ddi_pred = ddi_pred[:,1]
        dti_pred = dti_pred[:,1]
        #ddi_pred = ddi_pred
        #dti_pred =dti_pred
        y_label_ddi = y_label_ddi+label_ddi.flatten().tolist()
        y_label_dti = y_label_dti+label_dti.flatten().tolist()
        ddi_pred = ddi_pred.detach().cpu().numpy()
        dti_pred = dti_pred.detach().cpu().numpy()
        y_pred_ddi =  y_pred_ddi + ddi_pred.flatten().tolist()
        y_pred_dti =  y_pred_dti + dti_pred.flatten().tolist()
    if args.scenario == 'warm':
        y_label_dti = y_label_dti[:list2[idx]]
        y_pred_dti = y_pred_dti[:list2[idx]]
    elif args.scenario=='cold':
        y_label_ddi = y_label_ddi[:list2[idx]]
        y_pred_ddi = y_pred_ddi[:list2[idx]]
    test_ddi_acc, test_ddi_roc, test_ddi_pre, test_ddi_recall, test_ddi_aupr=_eval(y_pred_ddi,y_label_ddi)
    test_dti_acc, test_dti_roc, test_dti_pre, test_dti_recall, test_dti_aupr=_eval(y_pred_dti,y_label_dti)
    #import pdb;pdb.set_trace()
    print("Test DTI | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}".
              format(test_dti_acc, test_dti_roc, test_dti_pre, test_dti_recall, test_dti_aupr))
    print('Test DDI | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}'.format(
            test_ddi_acc, test_ddi_roc, test_ddi_pre, test_ddi_recall, test_ddi_aupr))
    return [test_ddi_acc, test_ddi_roc, test_ddi_aupr], [test_dti_acc, test_dti_roc, test_dti_aupr]

def get_features_ddi(data,fp_df,prodes_df,use_pro):
    drug_features = pd.merge(data,fp_df,how='left',left_on='head',right_on='drug_id').iloc[:,4:].values
    # for idx, drug in enumerate(drug_features):13
        # for i in range(1029):
            # if drug[i]==1.0:
                # import pdb;pdb.set_trace()
            # else:
                # print(drug[i])
    pro_features = pd.merge(data,prodes_df,how='left',left_on='tail',right_on='drug_id').iloc[:,4:].values
    if use_pro:
        feature = np.concatenate([drug_features,pro_features],axis=1)
    else:
        feature = drug_features
    return drug_features,pro_features
def get_features_hf(data,embed,ent_id):
    length = len(data)
    hf_features = []
    for i in range(length):
        head1 = data['head'][i]
        head2 = data['tail'][i]
        idx1  = ent_id[head1]
        idx2  = ent_id[head2]
        f1 = np.array(embed[idx1].cpu().detach().numpy())
        f2 = np.array(embed[idx2].cpu().detach().numpy())
        f= np.concatenate((f1,f2))
        hf_features.append(f)
    return np.array(hf_features)
    ####for 循环
def init_normal(m):
    #import pdb;pdb.set_trace()
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m,nn.Conv1d):
        nn.init.normal_(m.weight)
        nn.init.normal_(m.bias)

def predict(i,model,use_pro,patience):
    train_dti,train_dti_pos,test_dti,data_all,train_ddi,train_ddi_pos,test_ddi = load_data(i)
    ####structure features
    columns = ['head','relation','tail']
    re_train_all = train_dti[columns]
    re_test_all = test_dti[columns]
    train_label_dti = train_dti['label'].values
    test_label_dti = test_dti['label'].values
    train_sf_dti_d,train_sf_dti_p = get_features(re_train_all,fp_df,prodes_df,use_pro)
    test_sf_dti_d,test_sf_dti_p = get_features(re_test_all,fp_df,prodes_df,use_pro)
    
    re_train_all_ddi = train_ddi[columns]
    re_test_all_ddi = test_ddi[columns]
    train_label_ddi = train_ddi['label'].values
    test_label_ddi = test_ddi['label'].values
    train_sf_ddi_d1,train_sf_ddi_d2 = get_features_ddi(re_train_all_ddi,fp_df,fp_df,use_pro)
    test_sf_ddi_d1,test_sf_ddi_d2 = get_features_ddi(re_test_all_ddi,fp_df,fp_df,use_pro)
    #re_train_all_aft = re_train_all.append(re_train_all_ddi)
    #re_test_all_aft = re_test_all.append(re_test_all_ddi)
    re_train_all_aft = pd.concat([re_train_all,re_train_all_ddi])
    re_test_all_aft = pd.concat([re_test_all,re_test_all_ddi])
    ##heterogeneous features
    ####使用pykeen
    KGE_TRAIN_PATH = kge_path+'train_kg_'+str(i+1)+'.csv'
    KGE_TEST_PATH = kge_path+'test_kg_'+str(i+1)+'.csv'
    KGE_TEST_PATH_DDI = kge_path+'test_kg_'+str(i+1)+'_ddi.csv'
    
    
    
    KGE_COLD_PATH = kge_cold_path+'train_kg_'+str(i+1)+'.csv'
    tf =TriplesFactory.from_path(KGE_IDX_PATH)
    E_TO_I = tf.entity_to_id
    R_TO_I = tf.relation_to_id
    if args.scenario =='warm':
        training =TriplesFactory.from_path(KGE_TRAIN_PATH)
        E_TO_I = training.entity_to_id
        R_TO_I = training.relation_to_id
        testing = TriplesFactory.from_path(KGE_TEST_PATH,entity_to_id = E_TO_I,relation_to_id=R_TO_I)
        testing_ddi = TriplesFactory.from_path(KGE_TEST_PATH_DDI,entity_to_id = E_TO_I,relation_to_id=R_TO_I)
    elif args.scenario == 'cold':
        training =TriplesFactory.from_path(KGE_COLD_PATH,entity_to_id = E_TO_I,relation_to_id=R_TO_I)
        _,validation,testing = training.split([.8,0.199999,.000001])
        testing_ddi = testing
    print("ONLY ____________________________\n")
    print(KGE_NAME+'\n')
    print('-------------------\n')

    model_kge = torch.load(os.path.join(args.kge,str(i),'trained_model.pkl'))
    en_re = model_kge.entity_representations[0](indices=None)
    
    train_hf_dti = get_features_hf(re_train_all,en_re,E_TO_I)
    test_hf_dti = get_features_hf(re_test_all,en_re,E_TO_I)
    
    train_hf_ddi = get_features_hf(re_train_all_ddi,en_re,E_TO_I)
    test_hf_ddi = get_features_hf(re_test_all_ddi,en_re,E_TO_I)
    
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'drop_last': False}
    params_test = {'batch_size': args.batch_size,
              'shuffle': False,
              'drop_last': False}
    train_set = Data_Encoder(i,train_label_dti,train_label_ddi,\
                                train_hf_dti,train_sf_dti_d,train_sf_dti_p,\
                                train_hf_ddi,train_sf_ddi_d1,train_sf_ddi_d2,mode = 'train')
    test_set     = Data_Encoder(i,test_label_dti,test_label_ddi,\
                                test_hf_dti,test_sf_dti_d,test_sf_dti_p,\
                                test_hf_ddi,test_sf_ddi_d1,test_sf_ddi_d2,mode = 'test')
    train_generator = data.DataLoader(train_set,**params)
    test_generator  = data.DataLoader(test_set,**params)
    model_kgeunit =  torch.load(os.path.join(args.kgeunit,'KGE-UNIT_'+str(i)+".pth"),map_location = torch.device('cuda:0'))
    test_ddi_performance,test_dti_performance = test(test_generator, model_kgeunit,0,i)

    return test_ddi_performance,test_dti_performance
def main():

    device = 'cuda:{}'.format(args.gpu) if args.gpu>=0 else 'cpu'
    model = MultiTaskLoss(**config)
    use_cuda = args.gpu > 0 and torch.cuda.is_available()
    #import pdb;pdb.set_trace()
    if use_cuda:
        torch.cuda.set_device(0)
        model = model.cuda()
    loss_history=[]
    res_path = 'res/'+args.scenario+'/'+KGE_NAME+'.txt'
    fp = open(res_path,'w')
    for j in range(0,1):
        ROC_DTI = []
        PR_DTI = []
        ROC_DDI = []
        PR_DDI = []
        roc_dti_avg = 0
        pr_dti_avg = 0
        roc_ddi_avg = 0
        pr_ddi_avg = 0
        ROC=[]
        PR= []
        ROC_2 = []
        PR_2 = []
        roc_avg = 0
        pr_avg = 0
        roc_2_avg = 0
        pr_2_avg = 0
        for i in range(10):
            print(i)
            #roc_dti, pr_dti, roc_ddi,pr_ddi,roc,pr,roc_2,pr_2 = train(i,model,True,10)
            test_ddi_p, test_dti_p = predict(i,model,True,10)
            _,roc_ddi,pr_ddi = test_ddi_p
            _,roc_dti,pr_dti = test_dti_p
            roc_dti_avg += roc_dti
            pr_dti_avg += pr_dti
            roc_ddi_avg +=roc_ddi
            pr_ddi_avg +=pr_ddi
            
            ROC_DTI.append(roc_dti)
            PR_DTI.append(pr_dti)
            ROC_DDI.append(roc_ddi)
            PR_DDI.append(pr_ddi)
            
        print(roc_dti_avg,pr_dti_avg,roc_ddi_avg,pr_ddi_avg)
        print("-----------------\n")
        print(ROC_DTI)
        print(PR_DTI)
        print(ROC_DDI)
        print(PR_DDI)
        print(np.std(ROC_DTI))
        print(np.std(PR_DTI))
        print(np.std(ROC_DDI))
        print(np.std(PR_DDI))
        import pdb;pdb.set_trace()
        for zcc in range(0,10):
            fp.write(str(ROC_DTI[zcc])+'  '+str(PR_DTI[zcc])+'  '+str(ROC_DDI[zcc])+'  '+str(PR_DDI[zcc])+'\n')
    fp.close()
main()
import pdb;pdb.set_trace()
