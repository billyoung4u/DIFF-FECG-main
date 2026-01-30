# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:13:32 2021
@author: wangxu
"""
import matplotlib; 
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from sklearn import preprocessing

import scipy
import pyedflib
from scipy.signal import butter, filtfilt
import wfdb

from sklearn.model_selection import train_test_split
from config import cfg
from scipy.signal import savgol_filter

from scipy.signal import medfilt
import scipy as sc
import util
from sklearn.preprocessing import MinMaxScaler
import data_util as du


def NormalizeData(data):

    return scale(data, axis=1)




def get_max_abs(sig):  # 将信号放缩到-1到1
    max_abs = max(abs(max(sig)), abs(min(sig)))
    return sig / max_abs



def iirnotch(data, fs):
    w3, w4 = scipy.signal.iirnotch(50, 50, fs=fs)  # 50hz陷波滤波器
    w5, w6 = scipy.signal.iirnotch(60, 50, fs=fs)  # 60hz陷波滤波器
    
    data = scipy.signal.filtfilt(w3, w4, data, axis = 1)
    data = scipy.signal.filtfilt(w5, w6, data, axis = 1)
    return data

def remove_powerline(x,fs):
    nyq = 0.5 * fs
    on, off = 45,55
    low = on / nyq
    high = off / nyq
    b, a = scipy.signal.butter(4, [low, high], btype='bandstop', analog=False)
    y = scipy.signal.filtfilt(b, a, x,axis = 1)
    return y


def windowingSig(sig1, labels, windowSize=128,overlap = False):
    if overlap:
        sig1 = du.pad_audio(sig1,windowSize)
        signalsWindow1 = du.enframe(sig1,windowSize)
        labels = du.pad_audio(labels,windowSize)
        labelsWindow = du.enframe(labels,windowSize)
    else:
        signalLen = sig1.shape[1]
        signalsWindow1 = [sig1[:, int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize +1, windowSize)]
    #    signalsWindow2 = [sig2[:, int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize +1, windowSize)]
        labelsWindow = [labels[:, int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize +1, windowSize)]

    return signalsWindow1, labelsWindow#, signalsWindow2



#五折交叉验证，根据test_idx来决定：
def get_namelist(db, train=True, test_idx = 0,val_idx = 0):
    namelist = du.get_namelist(db)
    

        
    if train:
        new_list = []
        for i in range(len(namelist)):
            if test_idx == i or val_idx == i:
                continue
            new_list.append(namelist[i])
    else:
        new_list = [namelist[test_idx]]
        
    
    return new_list


    

def normalize_signal(signal):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True, clip= True)#
    
    signal = scaler.fit_transform(signal.T).T
    return signal


#txt = []
class FECGDataset(Dataset):
    def __init__(self, c,db='b2db',train=True, seg_len = 600,fs = 200,test_idx = 0, val_idx = -1,aecg_channel = [0],smooth= False,
                 fecg_label = False,mecg_label = False, remove_pl = False, train_all_channel = False):
        
        super(FECGDataset, self).__init__()
        
        self.data_path, self.contain_fecg, self.origin_fs = du.get_db_msg(db) 
        self.fs = fs
        self.db = db
        self.namelist = get_namelist(db, train, test_idx,val_idx)
        self.aecg_channel= aecg_channel
        self.test_idx = test_idx
        self.val_idx = val_idx
        if val_idx == -1:
            self.val_idx = self.test_idx
        
        self.train_all_channel = train_all_channel
        self.smooth = smooth
        self.fecg_label = fecg_label
        self.mecg_label = mecg_label
        self.remove_pl = remove_pl
        self.filter = c.filter
        self.record_len = 0
        self.config = c
        ecgWindows,labelWindows, self.fqrs_rpeaks = self.prepareData(train=train, seg_len = seg_len)
        labelWindows = normalize_signal(labelWindows.squeeze(axis=1))
        labelWindows= np.expand_dims(labelWindows,axis=1)
        self.X_train, self.Y_train = np.array(ecgWindows),np.array(labelWindows) #np.array(fecgWindows)
        
    
    def get_fqrs(self):
        return self.fqrs_rpeaks
    
    def get_record_len(self):
        return self.record_len
        
    def readData(self, name, aecg_channel):
        if self.fecg_label:
            if self.db == 'syn':
                data = du.get_data_from_syn(name, aecg_channel, return_fecg =True, config = self.config)
            else:
                data = du.get_data(self.db,name, aecg_channel, return_fecg =True)
            if self.contain_fecg:
                if self.db == 'syn':
                    ecg,mecg,fecg,peaks = data
                else:
                    ecg,fecg,peaks = data
            else:
                ecg,peaks = data
           
        else:
 
                ecg,peaks = du.get_data(self.db,name, aecg_channel, return_fecg = False)
       
        ecg = self.preprocess(ecg, self.origin_fs)
        
        scale_fecg_fs = self.origin_fs/self.fs
        peaks = np.asarray(np.floor_divide(peaks,scale_fecg_fs),'int64')
        
        
        
        if self.fecg_label and self.contain_fecg:
            fecg = self.preprocess(fecg, self.origin_fs)
           # fecg = normalize_signal(fecg)
            return ecg,ecg, fecg, peaks

        return ecg,ecg, ecg, peaks
#         else:
#             return abdECG, fetalECG
    
    def preprocess(self, signal, origin_fs):

        if self.filter is not None:
            signal = du.butter_bandpass_filter(signal, self.filter[0], self.filter[1], origin_fs)
    
       # signal = iirnotch(signal, origin_fs)
        if self.remove_pl:
            signal = remove_powerline(signal,origin_fs)
        else:
            signal = iirnotch(signal, origin_fs)
        signal =scale(signal, axis=1)
        scale_fs = origin_fs/self.fs
        if self.smooth:
            signal = savgol_filter(signal, origin_fs//20,9,axis = 1)
        signal = scipy.signal.resample(signal, int(signal.shape[1] / scale_fs), axis=1)

        return signal
    
    
    def prepareData(self, train=True, seg_len = 600):
        ecgAll, labels,peakAll= None, None,None
        overlap = True
        cnt = 0
        pad = False
   #     print(namelist)
        for name in self.namelist:
            for ac in self.aecg_channel:
                if self.fecg_label:
                    if self.train_all_channel:
                        ecg,mecg,fecg, peaks = self.readData(name,[ac])
                  #      print(ac)
                    else:
                        ecg,mecg,fecg, peaks = self.readData(name, self.aecg_channel)
                    ecg_len = ecg.shape[-1]

                    label = fecg
                else:
                    ecg, peaks = self.readData(name)
                    ecg_len = ecg.shape[1]






                if ecgAll is None:
                    ecgAll = ecg
                    #fecgAll = fecgDelayed
                    labels = label
                    peakAll = peaks

                else:
              #      print(ecgAll.shape)
                    ecgAll = np.append(ecgAll, ecg, axis=1)

                   # fecgAll = np.append(fecgAll, fecgDelayed, axis=1)
                    labels = np.append(labels, label, axis=1)
                    peakAll = np.append(peakAll, peaks+cnt*ecg_len)
            
                cnt += 1
                if self.train_all_channel is not True:
                    break
        self.record_len = ecgAll.shape[-1]
        #ecgAll = NormalizeData(ecgAll)
        ecgWindows, labelWindows = windowingSig(ecgAll, labels, windowSize=seg_len, overlap = overlap)
        ecgWindows = np.asarray(ecgWindows)
     #   fecgWindows = np.asarray(fecgWindows)
        labelWindows = np.asarray(labelWindows)

        return ecgWindows,labelWindows, peakAll
             

    
    
    
    

    def __getitem__(self, index):        
        dataset_x = self.X_train[index,:,:]
        dataset_y = self.Y_train[index,:,:]
        return dataset_x, dataset_y
        # return dataset_x, dataset_y, data_xx_idx

    def __len__(self):
        return self.X_train.shape[0]
    
  
