import numpy as np
import pandas as pd
import torch
from torch.utils import data



class Data_Encoder(data.Dataset):
    def __init__(self,cv_idx, labels_dti,labels_ddi,hf_dti,sf_dti_drug,sf_dti_protein,hf_ddi,sf_ddi_drug1,sf_ddi_drug2,mode= 'train'):
        print("___Data_Encoder_______\n")
        if mode == 'train':
            if len(labels_ddi)>len(labels_dti):
                self.labels_dti = np.hstack((labels_dti,labels_dti,labels_dti))[:len(labels_ddi)]
                self.labels_ddi = labels_ddi
                #self.hfs_dti = hf_dti.values[:,1:hf_dti.shape[1]]
                #self.hfs_dti = hf_dti.values
                self.hfs_dti = hf_dti
                self.hfs_dti = np.vstack((self.hfs_dti,self.hfs_dti,self.hfs_dti))[:len(labels_ddi)]
                self.sfs_dti_drug = np.vstack((sf_dti_drug,sf_dti_drug,sf_dti_drug))[:len(labels_ddi)]
                self.sfs_dti_protein = np.vstack((sf_dti_protein,sf_dti_protein,sf_dti_protein))[:len(labels_ddi)]
                #self.hfs_ddi = hf_ddi.values[:,1:hf_ddi.shape[1]]
                #self.hfs_ddi = hf_ddi.values
                self.hfs_ddi = hf_ddi
                self.sfs_ddi_drug1 = sf_ddi_drug1
                self.sfs_ddi_drug2 = sf_ddi_drug2
            else:
                self.labels_ddi = np.hstack((labels_ddi,labels_ddi,labels_ddi))[:len(labels_dti)]
                self.labels_dti = labels_dti
                #self.hfs_dti = hf_dti.values[:,1:hf_dti.shape[1]]
                #self.hfs_dti = hf_dti.values
                self.hfs_ddi = hf_ddi
                self.hfs_ddi = np.vstack((self.hfs_ddi,self.hfs_ddi,self.hfs_ddi))[:len(labels_dti)]
                self.sfs_ddi_drug1 = np.vstack((sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1))[:len(labels_dti)]
                self.sfs_ddi_drug2 = np.vstack((sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2))[:len(labels_dti)]
                #self.hfs_ddi = hf_ddi.values[:,1:hf_ddi.shape[1]]
                #self.hfs_ddi = hf_ddi.values
                self.hfs_dti = hf_dti
                self.sfs_dti_drug = sf_dti_drug
                self.sfs_dti_protein = sf_dti_protein
            print("---------train -----------\n")
            print("labels_dti shape:",self.labels_dti.shape)
            print("labels_ddi shape:",self.labels_ddi.shape)
            print("hfs_dti shape:",self.hfs_dti.shape)
            print("sfs_dti_drug shape:",self.sfs_dti_drug.shape)
            print("sfs_dti_protein shape:",self.sfs_dti_protein.shape)
            print("hfs_ddi shape:",self.hfs_ddi.shape)
            print("sfs_ddi_drug1 shape:",self.sfs_ddi_drug1.shape)
            print("sfs_ddi_drug2 shape:",self.sfs_ddi_drug2.shape)
        elif mode =='test':
            if len(labels_ddi)>len(labels_dti):
                self.labels_dti = np.hstack((labels_dti,labels_dti,labels_dti,labels_dti,labels_dti,labels_dti,labels_dti))[:len(labels_ddi)]
                self.labels_ddi = labels_ddi
                #self.hfs_dti = hf_dti.values[:,1:hf_dti.shape[1]]
                #self.hfs_dti = hf_dti.values
                self.hfs_dti = hf_dti
                self.hfs_dti = np.vstack((self.hfs_dti,self.hfs_dti,self.hfs_dti,self.hfs_dti,self.hfs_dti,self.hfs_dti,self.hfs_dti))[:len(labels_ddi)]
                self.sfs_dti_drug = np.vstack((sf_dti_drug,sf_dti_drug,sf_dti_drug,sf_dti_drug,sf_dti_drug,sf_dti_drug,sf_dti_drug))[:len(labels_ddi)]
                self.sfs_dti_protein = np.vstack((sf_dti_protein,sf_dti_protein,sf_dti_protein,sf_dti_protein,sf_dti_protein,sf_dti_protein,sf_dti_protein))[:len(labels_ddi)]
                #self.hfs_ddi = hf_ddi.values[:,1:hf_ddi.shape[1]]
                #self.hfs_ddi = hf_ddi.values
                self.hfs_ddi = hf_ddi
                self.sfs_ddi_drug1 = sf_ddi_drug1
                self.sfs_ddi_drug2 = sf_ddi_drug2
            else:
                self.labels_ddi = np.hstack((labels_ddi,labels_ddi,labels_ddi,labels_ddi,labels_ddi,labels_ddi,labels_ddi,labels_ddi,labels_ddi,labels_ddi,labels_ddi))[:len(labels_dti)]
                self.labels_dti = labels_dti
                #self.hfs_dti = hf_dti.values[:,1:hf_dti.shape[1]]
                #self.hfs_dti = hf_dti.values
                self.hfs_ddi = hf_ddi
                self.hfs_ddi = np.vstack((self.hfs_ddi,self.hfs_ddi,self.hfs_ddi,self.hfs_ddi,self.hfs_ddi,self.hfs_ddi,self.hfs_ddi))[:len(labels_dti)]
                self.sfs_ddi_drug1 = np.vstack((sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1,sf_ddi_drug1))[:len(labels_dti)]
                self.sfs_ddi_drug2 = np.vstack((sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2,sf_ddi_drug2))[:len(labels_dti)]
                #self.hfs_ddi = hf_ddi.values[:,1:hf_ddi.shape[1]]
                #self.hfs_ddi = hf_ddi.values
                self.hfs_dti = hf_dti
                self.sfs_dti_drug = sf_dti_drug
                self.sfs_dti_protein = sf_dti_protein
            print("---------test -----------\n")
            print("labels_dti shape:",self.labels_dti.shape)
            print("labels_ddi shape:",self.labels_ddi.shape)
            print("hfs_dti shape:",self.hfs_dti.shape)
            print("sfs_dti_drug shape:",self.sfs_dti_drug.shape)
            print("sfs_dti_protein shape:",self.sfs_dti_protein.shape)
            print("hfs_ddi shape:",self.hfs_ddi.shape)
            print("sfs_ddi_drug1 shape:",self.sfs_ddi_drug1.shape)
            print("sfs_ddi_drug2 shape:",self.sfs_ddi_drug2.shape)
    def __len__(self):
        return len(self.hfs_ddi)
    
    def __getitem__(self,index):
        #print(index)
        #index= self.IDs[index]
        label_dti = self.labels_dti[index]
        label_ddi = self.labels_ddi[index]
        hf_dti = self.hfs_dti[index]
        sf_dti_drug = self.sfs_dti_drug[index]
        sf_dti_protein = self.sfs_dti_protein[index]
        hf_ddi = self.hfs_ddi[index]
        sf_ddi_drug1 = self.sfs_ddi_drug1[index]
        sf_ddi_drug2 = self.sfs_ddi_drug2[index]
        return hf_dti,sf_dti_drug,sf_dti_protein,hf_ddi,sf_ddi_drug1,sf_ddi_drug2,label_dti,label_ddi