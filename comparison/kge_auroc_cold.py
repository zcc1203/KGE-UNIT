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
import numpy as np
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
torch.cuda.set_device(0)

curDirectory = os.getcwd()
print("curDirectory_new:", curDirectory)




data_path = './data/luo/data_folds/cold_start_1_1/'
data_path_ddi = './data/luo/data_folds/cold_start_1_1_ddi/'
SAVE_PATH = './run_comparison/cold/'

parser = ArgumentParser(description='KGE TRAINING.')
parser.add_argument('--kge', default='ConvE')
parser.add_argument('--save', action='store_true', help='only save final checkpoint')



args = parser.parse_args()
KGE_NAME = args.kge

with open(os.path.join('./comparison/models/cold/', KGE_NAME,'best_pipeline/pipeline_config.json'), 'r') as config_file:
    model_config = json.load(config_file)
import pdb;pdb.set_trace()
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



for yy in range(10):

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

    training,validation,testing = tf.split([.8,.000001,0.199999])
    print(yy)
    print(KGE_NAME)
    print("-------------------------\n")
    # result = pipeline(training=training,validation = validation,testing = validation,
                       # model=KGE_NAME,device = 'gpu',model_kwargs=dict(embedding_dim =400),
                                                                         # training_kwargs = dict(num_epochs=1,batch_size = 1024),
                                                                             # stopper = 'early',
                                                                         # stopper_kwargs= {"frequency":20,"patience":5,"relative_delta":0.005,"metric":"hits@10"},)
    result = pipeline(training=training,validation = validation,testing = validation,
                        model=model_config["pipeline"]["model"],
        model_kwargs=model_config["pipeline"]["model_kwargs"],
        loss=model_config["pipeline"]["loss"],
        loss_kwargs=model_config["pipeline"].get("loss_kwargs"),
        regularizer=model_config["pipeline"].get("regularizer"),
        optimizer=model_config["pipeline"]["optimizer"],
        optimizer_kwargs=model_config["pipeline"].get("optimizer_kwargs"),
        training_loop=model_config["pipeline"]["training_loop"],
        training_kwargs=model_config["pipeline"].get("training_kwargs"),
        negative_sampler=model_config["pipeline"].get("negative_sampler"),
        negative_sampler_kwargs=model_config["pipeline"].get("negative_sampler_kwargs"),
        stopper=model_config["pipeline"].get("stopper"),
        stopper_kwargs=model_config["pipeline"].get("stopper_kwargs"),
        evaluator=model_config["pipeline"].get("evaluator"),
        evaluator_kwargs=model_config["pipeline"].get("evaluator_kwargs"),
        evaluation_kwargs=model_config["pipeline"].get("evaluation_kwargs"),)

    if args.save:
        if not os.path.exists(os.path.join(SAVE_PATH,KGE_NAME)):
            os.makedirs(os.path.join(SAVE_PATH,KGE_NAME))
        result.save_to_directory(os.path.join(SAVE_PATH,KGE_NAME,str(yy)))
    #model2 = torch.load('*.pkl')
    #en_re = model2.entity_representations[0]
    model =result.model
    tftrain = result.training
    test = pd.read_csv(TEST_PATH)[['head','relation','tail','label']]
    test= shuffle(test)
    HEAD = test['head'].values
    TAIL = test['tail'].values
    test_score=[]
    flag= False
    #triples_factory= tftrain
    for h, t in zip(HEAD,TAIL):
        cc = predict.predict_target(model =model,head=h,tail = t,triples_factory= tftrain).df
        res = cc.values
        for i in range(len(res)):
            if res[i][2]=='drug_target_interaction':
                test_score.append(res[i][1])
                flag=True
        if flag ==False:
            test_score.append(0)
    test_label = test['label'].values

    roc = roc_auc(test_label,test_score)
    pr = pr_auc(test_label,test_score)
    ROC_DTI.append(roc)
    PR_DTI.append(pr)
    print("dti AUC:",roc)
    print("dti AUPR:",pr)

    test = pd.read_csv(TEST_PATH_DDI)[['head','relation','tail','label']]
    test= shuffle(test)
    HEAD = test['head'].values
    TAIL = test['tail'].values
    test_score=[]
    flag= False
    idx =0
    for h, t in zip(HEAD,TAIL):
        cc = predict.predict_target(model =model,head=h,tail = t,triples_factory= tftrain).df
        res = cc.values
        for i in range(len(res)):
            if res[i][2]=='DDI':
                test_score.append(res[i][1])
                flag=True
        if flag ==False:
            test_score.append(0)
        idx+=1
    test_label = test['label'].values

    roc = roc_auc(test_label,test_score)
    pr = pr_auc(test_label,test_score)
    print("dDi AUC:",roc)
    print("dDi AUPR:",pr)
    ROC_DDI.append(roc)
    PR_DDI.append(pr)
print("ROC_DTI:",ROC_DTI)
print("AUPR_DTI:",PR_DTI)
print("ROC_DDI:",ROC_DDI)
print("AUPR_DDI:",PR_DDI)
print("AVG_ROC_DTI:",avg(ROC_DTI),
      "AVG_AUPR_DTI:",avg(PR_DTI),
      "AVG_ROC_DDI:",avg(ROC_DDI),
      "AVG_AUPR_DDI:",avg(PR_DDI))

print("STD_ROC_DTI:",np.std(ROC_DTI))
print("STD_AUPR_DTI:",np.std(PR_DTI))
print("STD_ROC_DDI:",np.std(ROC_DDI))
print("STD_AUPR_DDI:",np.std(PR_DDI))
import pdb;pdb.set_trace()

