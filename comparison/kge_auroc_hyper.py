from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline 
from pykeen import predict
from pykeen.datasets.nations import NATIONS_TRAIN_PATH,NATIONS_TEST_PATH
from sklearn import metrics
from argparse import ArgumentParser
import csv
import pandas as pd
from sklearn.utils import shuffle
import pathlib
HERE = pathlib.Path(__file__).resolve().parent
print(HERE)
print(NATIONS_TRAIN_PATH)
import os
import torch
from pykeen.hpo.hpo import hpo_pipeline
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
torch.cuda.set_device(0)
import json

curDirectory = os.getcwd()
print("curDirectory_new:", curDirectory)




data_path = './data/luo/data_folds/warm_start_1_1/'
data_path_ddi = './data/luo/data_folds/warm_start_1_1_ddi/'
SAVE_PATH = 'comparison/models/'

parser = ArgumentParser(description='KGE TRAINING.')
parser.add_argument('--kge', default='MuRe')
parser.add_argument('--save', action='store_true', help='only save final checkpoint')



args = parser.parse_args()
KGE_NAME = args.kge
def avg(inp):
    sum_=sum(inp)
    num = len(inp)
    return sum_/num
def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

######

#@#####


ROC_DTI = []
PR_DTI = []
ROC_DDI = []
PR_DDI = []

with open(os.path.join('./comparison/json', f'{KGE_NAME}.json'), 'r') as config_file:
    model_config = json.load(config_file)
import pdb;pdb.set_trace()


for yy in range(1):

    TEST_PATH = os.path.join(data_path,'test_fold_'+str(yy+1)+'.csv')
    TEST_PATH_DDI = os.path.join(data_path_ddi,'test_fold_'+str(yy+1)+'.csv')
    TRAIN_PATH = os.path.join(data_path,'train_fold_'+str(yy+1)+'.csv')
    TRAIN_PATH_DDI = os.path.join(data_path_ddi,'train_fold_'+str(yy+1)+'.csv')
    
    INPUT_KG_PATH = './data/luo/kg_nodtiddi.csv'
    
    kg =  pd.read_csv(INPUT_KG_PATH,delimiter='\t')[['head','relation','tail']]
    train_dti = pd.read_csv(TRAIN_PATH)[['head','relation','tail','label']]
    train_dti_pos = train_dti[train_dti['label']==1][['head','relation','tail']]
    train_ddi = pd.read_csv(TRAIN_PATH_DDI)[['head','relation','tail','label']]
    train_ddi_pos = train_ddi[train_ddi['label']==1][['head','relation','tail']]

    data = pd.concat([train_dti_pos,train_ddi_pos,kg])
    OUTPUT_KG_PATH = 'kg_train_all.csv'
    data.to_csv(OUTPUT_KG_PATH,header=None,index=False,sep='\t')
    tf = TriplesFactory.from_path(OUTPUT_KG_PATH)

    training,validation = tf.split([.8,.2])
    print(yy)
    print(KGE_NAME)
    print("-------------------------\n")
    # result = pipeline(training=training,validation = validation,testing = validation,
                       # model=KGE_NAME,device = 'gpu',model_kwargs=dict(embedding_dim =400),
                                                                         # training_kwargs = dict(num_epochs=1,batch_size = 1024),
                                                                             # stopper = 'early',
                                                                         # stopper_kwargs= {"frequency":20,"patience":5,"relative_delta":0.005,"metric":"hits@10"},)
    #result = pipeline(training=training,validation = validation,testing = validation,
    #                   model=KGE_NAME,device = 'gpu',model_kwargs=dict(embedding_dim =400),
    #                                                                     training_kwargs = dict(num_epochs=150,batch_size = 1024),)

    result = hpo_pipeline(dataset=None,training= training,validation = validation,testing = validation, 
        model=model_config["pipeline"]["model"],
        model_kwargs=model_config["pipeline"]["model_kwargs"],
        model_kwargs_ranges=model_config["pipeline"].get("model_kwargs_ranges"),
        loss=model_config["pipeline"]["loss"],
        loss_kwargs=model_config["pipeline"].get("loss_kwargs"),
        loss_kwargs_ranges=model_config["pipeline"].get("loss_kwargs_ranges"),
        regularizer=model_config["pipeline"].get("regularizer"),
        optimizer=model_config["pipeline"]["optimizer"],
        optimizer_kwargs=model_config["pipeline"].get("optimizer_kwargs"),
        optimizer_kwargs_ranges=model_config["pipeline"].get("optimizer_kwargs_ranges"),
        training_loop=model_config["pipeline"]["training_loop"],
        training_kwargs=model_config["pipeline"].get("training_kwargs"),
        training_kwargs_ranges=model_config["pipeline"].get("training_k+wargs_ranges"),
        negative_sampler=model_config["pipeline"].get("negative_sampler"),
        negative_sampler_kwargs=model_config["pipeline"].get("negative_sampler_kwargs"),
        negative_sampler_kwargs_ranges=model_config["pipeline"].get("negative_sampler_kwargs_ranges"),
        stopper=model_config["pipeline"].get("stopper"),
        stopper_kwargs=model_config["pipeline"].get("stopper_kwargs"),
        evaluator=model_config["pipeline"].get("evaluator"),
        evaluator_kwargs=model_config["pipeline"].get("evaluator_kwargs"),
        evaluation_kwargs=model_config["pipeline"].get("evaluation_kwargs"),
        n_trials=model_config["optuna"]['n_trials'],
        timeout=model_config["optuna"]["timeout"],
        metric=model_config["optuna"]["metric"],
        direction=model_config["optuna"]["direction"],
        sampler=model_config["optuna"]["sampler"],
        pruner=model_config["optuna"]["pruner"],)
    if args.save:
        if not os.path.exists(os.path.join(SAVE_PATH,KGE_NAME)):
            os.makedirs(os.path.join(SAVE_PATH,KGE_NAME))
        result.save_to_directory(os.path.join(SAVE_PATH,KGE_NAME))

import pdb;pdb.set_trace()

