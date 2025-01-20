import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from librosa import effects
import soundfile as sf
import random
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav

___author__ = "Hemlata Tak, Massimiliano Todisco"
__email__ = "{tak,todisco}@eurecom.fr"


#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature


# def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    
#     d_meta = {}
#     file_list=[]
#     with open(dir_meta, 'r') as f:
#          l_meta = f.readlines()

#     if (is_train):
#         for line in l_meta:
#              _, key,_,_,label = line.strip().split(' ')
#              file_list.append(key)
#              '''key : audio_file name'''
#              d_meta[key] = 1 if label == 'bonafide' else 0
#         return d_meta,file_list
    
#     elif(is_eval):
#         for line in l_meta:
#             key= line.strip()
#             file_list.append(key)
#         return file_list
#     else:
#         for line in l_meta:
#              _, key,_,_,label = line.strip().split(' ')
#              file_list.append(key)
#              d_meta[key] = 1 if label == 'bonafide' else 0
#         return d_meta,file_list
'''eval change'''
def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    
    # d_meta = {}
    label_list=[]
    variant_list=[]
    random_list=[]
    src_list=[]
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    # if (is_train):
    #     for line in l_meta:
    #          _, key,_,_,label = line.strip().split(' ')
    #          file_list.append(key)
    #          '''key : audio_file name'''
    #          d_meta[key] = 1 if label == 'bonafide' else 0
    #     return d_meta,file_list
    
    # elif(is_eval):
    for line in l_meta:
        variant,path,random,src,fake=line.strip().split(' ')
        variant_list.append(variant)
        
        file_list.append(path)
        random_list.append(random)
        src_list.append(src)
        label_list.append(fake)
    return variant_list,file_list,random_list,src_list,label_list
    # else:
    #     for line in l_meta:
    #          _, key,_,_,label = line.strip().split(' ')
    #          file_list.append(key)
    #          d_meta[key] = 1 if label == 'bonafide' else 0
    #     return d_meta,file_list



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir,algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=64600 # take ~4 sec audio (64600 samples)
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
            
        key = self.list_IDs[index]
        X,fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000) 
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_pad= pad(Y,self.cut)
        x_inp= Tensor(X_pad)
        y = self.labels[key]
        
        return x_inp, y
class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, variant_IDs,list_IDs,random_IDs,src_IDs,fake_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),'''
               
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.variant_IDs=variant_IDs
        self.random_IDs=random_IDs
        self.src_IDs=src_IDs
        self.fake_IDs=fake_IDs
        self.cut=64600 # take ~4 sec audio (64600 samples)
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
            
        key = self.list_IDs[index]
        variant = self.variant_IDs[index]
        random = self.random_IDs[index]
        src = self.src_IDs[index]
        label=1 if self.fake_IDs[index]=='bonafide' else 0
        # X, fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
        X,fs=sf.read(os.path.join(self.base_dir,key))
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp,key,variant,random,src,label          


