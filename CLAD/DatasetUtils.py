'''
Datasets and augmentations used in CLAD. This code is built on RawNet2 and ASVspoof 2021 Baseline repository.
'''

import os
import random
import torch
# import torchaudio
# import TorchTimeStretch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import librosa  # used by ASVspoof official data utils
import soundfile as sf




# Evaluation utilities
# Mainly modified from https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/data_utils.py by "Hemlata Tak"
# Obtain speaker information for SAMO implementation
def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    # utt2spk = {}
    # d_meta = {}
    variant_list=[]
    file_list=[]
    random_flag_list=[]
    src_list=[]
    label_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            # spk, key,_,_,label = line.strip().split(' ')
            variant, key,random_flag,src,label = line.strip().split(' ')
            # utt2spk[key] = spk
            variant_list.append(variant)
            file_list.append(key)
            random_flag_list.append(random_flag)
            src_list.append(src)
            label = 1 if label == 'bonafide' else 0
            label_list.append(label)
        # return d_meta,file_list,utt2spk
        return variant_list,file_list,random_flag_list,src_list,label_list
    
    elif(is_eval):
        for line in l_meta:
            # spk, key,_,_,label = line.strip().split(' ')
            variant, key,random_flag,src,label = line.strip().split(' ')
            # utt2spk[key] = spk
            variant_list.append(variant)
            file_list.append(key)
            random_flag_list.append(random_flag)
            src_list.append(src)
            label = 1 if label == 'bonafide' else 0
            label_list.append(label)
        # return d_meta,file_list,utt2spk
        return variant_list,file_list,random_flag_list,src_list,label_list
        # for line in l_meta:
        #     key= line.strip()
        #     file_list.append(key)
        # return file_list
    else: 
        # for line in l_meta:
        #     spk, key,_,_,label = line.strip().split(' ')
        #     utt2spk[key] = spk
        #     file_list.append(key)
        #     d_meta[key] = 1 if label == 'bonafide' else 0
        # return d_meta,file_list,utt2spk
        for line in l_meta:
            # spk, key,_,_,label = line.strip().split(' ')
            variant, key,random_flag,src,label = line.strip().split(' ')
            # utt2spk[key] = spk
            variant_list.append(variant)
            file_list.append(key)
            random_flag_list.append(random_flag)
            src_list.append(src)
            label = 1 if label == 'bonafide' else 0
            label_list.append(label)
        # return d_meta,file_list,utt2spk
        return variant_list,file_list,random_flag_list,src_list,label_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
    # def __init__(self, list_IDs, labels, utt2spk, base_dir, cut_length=64600):
    def __init__(self, list_IDs, label_IDs, random_IDs,src_IDs,variant_IDs, base_dir, cut_length=64600):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            # self.list_IDs = list_IDs
            # self.labels = label_ID
            # self.base_dir = base_dir
            # self.cut = cut_length
            # self.utt2spk = utt2spk
            self.list_IDs=list_IDs
            self.label_IDs=label_IDs
            self.random_IDs=random_IDs
            self.variant_IDs=variant_IDs
            # self.variant_IDs=variant_IDs
            self.src_IDs=src_IDs
            self.base_dir = base_dir
            self.cut=cut_length
        
    def __len__(self):
           return len(self.list_IDs)


    def __getitem__(self, index):
            # self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            # X,fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000) 
            X,fs=sf.read(os.path.join(self.base_dir,key))
            X_pad= pad(X,self.cut)
            x_inp= torch.Tensor(X_pad)
            x_inp = torch.unsqueeze(x_inp, 0)  # added by haulyn5, add a dimension for channels In order to be consistent with the previous dataset
            y = self.label_IDs[index]
            variant = self.variant_IDs[index]
            src=self.src_IDs[index]
            random_label=self.random_IDs[index]
            return x_inp, y,key,variant,src,random_label

# A upgraded version of pad_or_clip function, which can process batched audio, zero padding or  clipping them to the same length
def pad_or_clip_batch(audio, audio_len, random_clip=True):
        '''
        Pad or randomly clip the audio to make it of length audio_len
        '''
        if audio.shape[-1] < audio_len:
            audio = torch.nn.functional.pad(audio, (0, audio_len - audio.shape[-1]))
        elif audio.shape[-1] > audio_len:
            if random_clip == True:
                # randomly clip the audio
                start = random.randint(0, audio.shape[-1] - audio_len)
            else:
                start = 0 # clip from the beginning, which is the standard implementation of AASIST and RawNet2
                audio = audio[:, start:start+ audio_len]
        return audio