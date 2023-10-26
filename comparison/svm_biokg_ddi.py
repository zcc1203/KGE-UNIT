################################
#This script provide the detailed complement of RF(baseline) and KGE_RF(discussed in paper 3.2.2)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import ampligraph as ampligraph
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import train_test_split_no_unseen,generate_corruptions_for_fit 
from ampligraph.latent_features import ComplEx,TransE,DistMult
import tensorflow as tf
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.utils import save_model,restore_model
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from thundersvm import SVC
#Des
################################################################
mms = MinMaxScaler(feature_range=(0,1))

#descriptors preparation
fp_id = pd.read_csv('./data/BioKG/comp_struc.csv')['drug_id']
df_proseq = pd.read_csv('./data/BioKG/pro_seq.csv')
#df_proseq.columns = ['pro_id','pro_ids','seq']
df_proseq.columns = ['pro_id','seq']
pro_id = df_proseq['pro_id']
drug_feats = np.loadtxt('./data/BioKG/fp_df.csv',skiprows=1,dtype=str,delimiter=',')
pro_feats = np.loadtxt('./data/BioKG/prodes_df.csv',skiprows=1,dtype=str,delimiter=',')
drug_feats = drug_feats[:,:-1].astype(float)
pro_feats = pro_feats[:,:-1].astype(float)

pro_feats_scaled = mms.fit_transform(pro_feats)
pro_feats_scaled2 = PCA(n_components=100).fit_transform(pro_feats_scaled)
pro_feats_scaled3 = mms.fit_transform(pro_feats_scaled2)

drug_feats_scaled = mms.fit_transform(drug_feats)
drug_feats_scaled2 = PCA(n_components=100).fit_transform(drug_feats_scaled)
drug_feats_scaled3 = mms.fit_transform(drug_feats_scaled2)
fp_df = pd.concat([fp_id,pd.DataFrame(drug_feats_scaled3)],axis=1)
prodes_df = pd.concat([pro_id,pd.DataFrame(pro_feats_scaled3)],axis=1)
import pdb;pdb.set_trace()
#Function
################################################################

def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def get_scaled_embeddings(model,train_triples,test_triples,get_scaled,n_components):
    [train_sub_embeddings,test_sub_embeddings] = [model.get_embeddings(x['head'].values, embedding_type='entity') for x in [train_triples,test_triples]]
    [train_obj_embeddings,test_obj_embeddings] = [model.get_embeddings(x['tail'].values, embedding_type='entity') for x in [train_triples,test_triples]]
    train_feats = np.concatenate([train_sub_embeddings,train_obj_embeddings],axis=1)
    test_feats = np.concatenate([test_sub_embeddings,test_obj_embeddings],axis=1)
    train_dense_features = mms.fit_transform(train_feats)
    test_dense_features = mms.transform(test_feats)
    if get_scaled:
        pca = PCA(n_components=n_components)
        scaled_train_dense_features = pca.fit_transform(train_dense_features)
        scaled_pca_test_dense_features = pca.transform(test_dense_features)
    else:
        scaled_train_dense_features = train_dense_features
        scaled_pca_test_dense_features = test_dense_features
    return scaled_train_dense_features,scaled_pca_test_dense_features


def get_features(data,fp_df,prodes_df,use_pro):
    drug_features_b = pd.merge(data,fp_df,how='left',left_on='head',right_on='drug_id')
    #import pdb;pdb.set_trace()
    drug_features = drug_features_b.iloc[:,4:105].values
    pro = pd.merge(data,fp_df,how='left',left_on='tail',right_on='drug_id')
    pro_features = pro.iloc[:,4:105].values
    if use_pro:
        feature = np.concatenate([drug_features,pro_features],axis=1)
    else:
        feature = drug_features
    return feature


def train(i):
    train = pd.read_csv('./data/BioKG/data_folds/warm_start_1_1_ddi/train_fold_'+str(i+1)+'.csv')
    test = pd.read_csv('./data/BioKG/data_folds/warm_start_1_1_ddi/test_fold_'+str(i+1)+'.csv')
    #model = restore_model(model_name_path='./eg_model/dismult_400_warm_1_10.pkl')   
    columns = ['head','relation','tail']
    #test_score = model.predict(test[columns])
    
    #kge performance evaluation
    #print(roc_auc(test_label,test_score))
    #print(pr_auc(test_label,test_score))
    #pre
    train = train
    test = test
    test_label = test['label'].values
    #import pdb;pdb.set_trace()
    re_train_all = train[columns]
    re_test_all = test[columns]
    #train_dense_features,test_dense_features = get_scaled_embeddings(model,re_train_all,re_test_all,get_scaled=False,n_components=50)
    pca = PCA(n_components=500)
    #train_dense_features_scaled = pca.fit_transform(train_dense_features)
    #test_dense_features_scaled = pca.transform(test_dense_features)
    train_des = get_features(re_train_all,fp_df,prodes_df,use_pro=True)
    test_des = get_features(re_test_all,fp_df,prodes_df,use_pro=True)
    #train_all_feats = np.concatenate([train_dense_features_scaled,train_des],axis=1)
    #test_all_feats = np.concatenate([test_dense_features_scaled,test_des],axis=1)
    train_label = train['label']
    #rf
    # clf_svm = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo',random_state=None)
    # clf_svm.fit(train_des,train_label)
    # pred_svm = clf_svm.predict(test_des)
    # roc = roc_auc(test_label,pred_svm)
    # pr = pr_auc(test_label,pred_svm)
    # print("roc_svm:",roc)
    # print("pr_svm",pr)
    # clf = RandomForestClassifier(n_estimators=50,
                                # criterion='entropy',
                                # random_state=0,
                                # class_weight='balanced',
                                # n_jobs=-1)
    # clf.fit(train_des,train_label)
    clf = GradientBoostingClassifier()
    clf.fit(train_des,train_label)
    pred = clf.predict_proba(test_des)
    roc = roc_auc(test_label,pred[:,1])
    pr = pr_auc(test_label,pred[:,1])
    print("roc:",roc)
    print("pr",pr)
    #kge_rf
    # clf = RandomForestClassifier(n_estimators=500,
                                # criterion='entropy',
                                #max_depth=50,
                                # random_state=0,
                                # class_weight='balanced',
                                # n_jobs=-1)
    # clf.fit(train_all_feats,train_label)
    # pred = clf.predict_proba(test_all_feats)
    # roc_s = roc_auc(test_label,pred[:,1])
    # pr_s = pr_auc(test_label,pred[:,1])
    # print(roc_s)
    # print(pr_s)
    #return roc,pr,roc_s,pr_s
    return roc,pr

###train
for i in range(10):
    print(i)
    roc,pr= train(i)
