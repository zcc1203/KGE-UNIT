from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)




class KGE_UNIT(nn.Sequential):
    def __init__(self, **config):
        super(KGE_UNIT, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        #self.hidden_dim_hf = config["hidden_dim_hf"]
        self.hidden_dim_hf = self.hidden_dim*2
        embedding_dim = config['embed_dim']
        #dim_ = self.hidden_dim*2+self.hidden_dim_hf
        dim_ = embedding_dim
        #import pdb;pdb.set_trace()
        self.dim_ =dim_
        self.task_num= config['task_num']
        #self.protein_layers= [self.hidden_dim,256,128,self.hidden_dim]
        #self.drug_layers = [self.hidden_dim,256,128,self.hidden_dim]
        self.protein_layers= [self.hidden_dim,256,self.hidden_dim]
        self.drug_layers = [self.hidden_dim,256,self.hidden_dim]
        #self.protein_layers = [dim_,64,self.hidden_dim]
        #self.drug_layers = [dim_,64,self.hidden_dim]
        self.hf_dti_layers= [800,512,256,self.hidden_dim_hf]
        self.hf_ddi_layers= [800,512,256,self.hidden_dim_hf]
        
        print("NUM HEAD = 4\n")
        print("embedding 32 \n")
        self.drug_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.drug_layers[i], out_channels=self.drug_layers[i+1],
                                                 kernel_size=3, padding=1) for i in range(len(self.drug_layers)-1)])
        self.protein_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.protein_layers[i], out_channels=self.protein_layers[i+1],
                                        kernel_size=3, padding=1) for i in range(len(self.protein_layers)-1)])
        self.hf_dti_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.hf_dti_layers[i], out_channels=self.hf_dti_layers[i+1],
                                        kernel_size=3, padding=1) for i in range(len(self.hf_dti_layers)-1)])
        self.hf_ddi_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.hf_ddi_layers[i], out_channels=self.hf_ddi_layers[i+1],
                                        kernel_size=3, padding=1) for i in range(len(self.hf_ddi_layers)-1)])
        self.task_fusion = nn.MultiheadAttention(embed_dim = dim_,num_heads=4,dropout= 0.1)
        self.smlp  = nn.Sequential(nn.Linear(dim_,dim_),nn.LayerNorm(dim_))
        self.smlp2 = nn.ModuleList([nn.Sequential(nn.Linear(dim_, dim_), nn.LayerNorm(dim_))  for t in range (self.task_num)])
        self.task_querys = nn.ModuleList([nn.MultiheadAttention(embed_dim=dim_, num_heads=4, dropout=0.1) for t in range(self.task_num)])
        self.mix_layers= [512,256,256]
        self.f_dti_mix = nn.ModuleList([nn.Conv1d(in_channels=self.mix_layers[i], out_channels=self.mix_layers[i+1],
                                                 kernel_size=3, padding=1) for i in range(len(self.mix_layers)-1)])
        self.f_ddi_mix = self.f_dti_mix
        #self.fc_hidden_dim = [512,256,128,64,32]
        #self.fc_hidden_dim = [256,64,32]
        #self.fc_hidden_dim = [self.hidden_dim_hf, 32,16,4]  ##不用flatten
        #self.fc_hidden_dim = [self.hidden_dim_hf*32,self.hidden_dim_hf,64, 16,4]
        self.fc_hidden_dim = [self.hidden_dim_hf*self.dim_,16]
        self.fc_layers=nn.ModuleList()
        for i in range(len(self.fc_hidden_dim)):
            if i == len(self.fc_hidden_dim)-1:
                self.fc_layers.append(nn.Linear(self.fc_hidden_dim[i], 2))
                #self.fc_layers.append(nn.Sigmoid())
            else:
                self.fc_layers.append(
                    nn.Linear(self.fc_hidden_dim[i], self.fc_hidden_dim[i+1]))
                #self.fc_layers.append(nn.ReLU())
                #self.fc_layers.append(nn.Dropout(dropout_prob))
        self.fc_layers_ddi = self.fc_layers
        self.fc_layers_2 = nn.ModuleList()
        #for i in range(len(self.fc_hidden_dim)):
        #    if 
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_2cls = nn.CrossEntropyLoss()
        #import pdb;pdb.set_trace()
    def forward(self,hf_dti,sf_dti_drug,sf_dti_protein,hf_ddi,sf_ddi_d1,sf_ddi_d2):
        drug_feats = sf_dti_drug.unsqueeze(2).repeat(1,1,self.dim_ ).cuda().float()
        for layer in self.drug_cnn:
            layer_feats = layer(drug_feats)
            #import pdb;pdb.set_trace()
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            drug_feats = F.leaky_relu(bn(layer_feats.cpu()).cuda())
        #protein_feats = self.embed_protein(sf_dti_protein.long().cuda())
        protein_feats = sf_dti_protein.unsqueeze(2).repeat(1,1,self.dim_ ).cuda().float()
        for layer in self.protein_cnn:
            layer_feats = layer(protein_feats)
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            protein_feats = F.leaky_relu(bn(layer_feats.cpu()).cuda())
        sf_feats_dti = torch.cat([drug_feats,protein_feats],dim=1)
        #import pdb;pdb.set_trace()
        #drug1_feats_ddi = self.embed_d1_ddi(sf_ddi_d1.long().cuda())
        #drug2_feats_ddi = self.embed_d2_ddi(sf_ddi_d2.long().cuda())
        drug1_feats_ddi = sf_ddi_d1.unsqueeze(2).repeat(1,1,self.dim_ ).cuda().float()
        drug2_feats_ddi = sf_ddi_d2.unsqueeze(2).repeat(1,1,self.dim_ ).cuda().float()
        for layer in self.drug_cnn:
            layer_feats = layer(drug1_feats_ddi)
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            drug1_feats_ddi = F.leaky_relu(bn(layer_feats.cpu()).cuda())
        for layer in self.drug_cnn:
            layer_feats = layer(drug2_feats_ddi)
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            drug2_feats_ddi = F.leaky_relu(bn(layer_feats.cpu()).cuda())
        sf_feats_ddi = torch.cat([drug1_feats_ddi,drug2_feats_ddi],dim=1)
        #hf_feats_dti = self.embed_hf_dti(hf_dti.long().cuda())
        #hf_feats_ddi = self.embed_hf_ddi(hf_ddi.long().cuda())
        hf_feats_dti = hf_dti.unsqueeze(2).repeat(1,1,self.dim_ ).cuda().float()
        hf_feats_ddi = hf_ddi.unsqueeze(2).repeat(1,1,self.dim_ ).cuda().float()
        for layer in self.hf_dti_cnn:
            layer_feats = layer(hf_feats_dti)
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            hf_feats_dti = F.leaky_relu(bn(layer_feats.cpu()).cuda())
        for layer in self.hf_ddi_cnn:
            layer_feats = layer(hf_feats_ddi)
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            hf_feats_ddi =  F.leaky_relu(bn(layer_feats.cpu()).cuda())
        feats_ddi0 = torch.cat([sf_feats_ddi,hf_feats_ddi],dim=1)
        feats_dti0 = torch.cat([sf_feats_dti,hf_feats_dti],dim=1)
        feats_ddi = sf_feats_ddi+hf_feats_ddi
        feats_dti = sf_feats_dti +hf_feats_dti
        for layer in self.f_ddi_mix:
            layer_feats = layer(feats_ddi0)
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            feats_ddi0 = F.leaky_relu(bn(layer_feats.cpu()).cuda())
        for layer in self.f_dti_mix:
            layer_feats = layer(feats_dti0)
            bn = nn.BatchNorm1d(layer_feats.shape[1])
            feats_dti0 = F.leaky_relu(bn(layer_feats.cpu()).cuda())
        feats_ddi1 = torch.flatten(feats_ddi,start_dim=1,end_dim=2)
        feats_dti1 = torch.flatten(feats_dti,start_dim=1,end_dim=2)
        feats_ddi1 = torch.unsqueeze(feats_ddi1,1)
        feats_dti1 = torch.unsqueeze(feats_dti1,1)
        task_cat1 = torch.cat([feats_ddi1,feats_dti1],dim=1)
        feats_ddi = feats_ddi0
        feats_dti = feats_dti0
        task_cat = torch.cat([feats_ddi,feats_dti],dim=1)
        task_cat = nn.LayerNorm(self.dim_)(task_cat.cpu()).cuda()
        task_cat = self.task_fusion(task_cat,task_cat,task_cat)[0]
        task_cat = self.smlp(task_cat)
        task_dti = self.smlp2[1](self.task_querys[1](feats_dti,task_cat,task_cat)[0])
        task_ddi = self.smlp2[0](self.task_querys[0](feats_ddi,task_cat,task_cat)[0])
        pred_dti = feats_dti+0.001*task_dti
        pred_ddi = feats_ddi+0.001*task_ddi
        
        pred_dti = torch.flatten(pred_dti,start_dim = 1,end_dim =2)
        pred_ddi = torch.flatten(pred_ddi,start_dim = 1,end_dim =2)
        for fc in self.fc_layers:
            fc_dti = fc(pred_dti)
            bn = nn.BatchNorm1d(fc_dti.shape[1])
            pred_dti =  F.leaky_relu(bn(fc_dti.cpu()).cuda())
        for fc in self.fc_layers_ddi:
            fc_ddi = fc(pred_ddi)
            bn = nn.BatchNorm1d(fc_ddi.shape[1])
            pred_ddi = F.leaky_relu(bn(fc_ddi.cpu()).cuda())
        #pred_dti = F.sigmoid(pred_dti)
        #pred_ddi = F.sigmoid(pred_ddi)
        #import pdb;pdb.set_trace()
        return pred_dti,pred_ddi
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        #import pdb;pdb.set_trace()
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class MultiTaskLoss(nn.Module):
    def __init__(self,**config):
        super(MultiTaskLoss,self).__init__()
        self.task_num = config['task_num']
        self.log_var = nn.Parameter(torch.zeros(self.task_num,requires_grad = True))
        self.tasks = KGE_UNIT(**config)
        self.criterion_2cls = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.smoothloss= LabelSmoothingCrossEntropy()
    def loss(self,dti_loss,ddi_loss,mode = 'weighted'):
        if mode =='weighted':
            pre1=torch.exp(-self.log_var[0])    
            pre2=torch.exp(-self.log_var[1])
            loss=torch.sum(pre1 * ddi_loss + self.log_var[0], -1)
            loss+=torch.sum(pre2 * dti_loss + self.log_var[1], -1)
            loss=torch.mean(loss)
            return loss
        elif mode=='single':
            #import pdb;pdb.set_trace()
            #loss=torch.sum(torch.cat(ddi_loss, dti_loss))
            loss = ddi_loss+dti_loss
            return loss
    def forward(self,hf_dti,sf_dti_drug,sf_dti_protein,hf_ddi,sf_ddi_d1,sf_ddi_d2,dti_labels=None,ddi_labels=None,mode= 'single',eval=False):
        if not eval:
            dti_pred,ddi_pred = self.tasks(hf_dti,sf_dti_drug,sf_dti_protein,hf_ddi,sf_ddi_d1,sf_ddi_d2)
            #print(dti_pred)
            #print(ddi_pred)
            #dti_pred = dti_pred.view(dti_pred.size(0))
            #ddi_pred = ddi_pred.view(ddi_pred.size(0))
            #ddi_loss = F.binary_cross_entropy(ddi_pred.float().cuda(),ddi_labels.float().cuda())
            ddi_one_hot = ddi_labels.numpy()
            ddi_one_hot = (np.arange(ddi_one_hot.max()+1)==ddi_one_hot[:,None]).astype(dtype='float32').squeeze()
            
            dti_one_hot = dti_labels.numpy()
            dti_one_hot = (np.arange(dti_one_hot.max()+1)==dti_one_hot[:,None]).astype(dtype='float32').squeeze()
            ddi_one_hot = ddi_one_hot.astype(np.int64)
            dti_one_hot = dti_one_hot.astype(np.int64)
            #import pdb;pdb.set_trace()
            #import pdb;pdb.set_trace()
            #使用nn.CrossEntropyLoss 输出2，训练不需要softmax,函数内嵌；测试需要加softmax
            ddi_loss = self.criterion_2cls(ddi_pred.cuda(),ddi_labels.cuda())
            dti_loss =self.criterion_2cls(dti_pred.cuda(),dti_labels.cuda())
            #使用F.binary_cross_entropy输出为1
            #dti_loss = F.binary_cross_entropy(dti_pred.float().cuda().squeeze(),dti_labels.float().cuda())
            #ddi_loss = F.binary_cross_entropy(ddi_pred.float().cuda().squeeze(),ddi_labels.float().cuda())
            #使用smoothl1loss 输出2，训练不需要softmax,函数内嵌；测试需要加softmax
            #ddi_loss = self.smoothloss(ddi_pred.cuda(),ddi_labels.cuda())
            #dti_loss= self.smoothloss(dti_pred.cuda(),dti_labels.cuda())
            loss = self.loss(dti_loss,ddi_loss,mode=mode)
            #loss = ddi_loss
            return loss,dti_loss,ddi_loss,dti_pred,ddi_pred
        else:
            dti_pred,ddi_pred = self.tasks(hf_dti,sf_dti_drug,sf_dti_protein,hf_ddi,sf_ddi_d1,sf_ddi_d2)
            dti_pred_softmax = torch.softmax(dti_pred,dim=1)
            ddi_pred_softmax = torch.softmax(ddi_pred,dim=1)
            return dti_pred_softmax,ddi_pred_softmax