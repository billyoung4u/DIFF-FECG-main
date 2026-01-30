from ecgdetectors import Detectors
import time,os
import numpy as np
import torch
import random
from torch.optim import lr_scheduler
import math
import scipy.stats as stats
from sklearn.preprocessing import scale
from scipy.signal import correlate
from scipy.signal import spectrogram
from sklearn.metrics import mean_squared_error
#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)



def ignore_side(r_ref, r_ans,fs,length,ignore_sec = 10):
    new_ref = []
    new_ans = []
    on,off = ignore_sec*fs, length-ignore_sec*fs
    for i in range(len(r_ref)):
        #ref = r_ref[i]
        tmp = []
        for j in range(len(r_ref[i])):
            if r_ref[i][j]>on and r_ref[i][j] < off:
                tmp.append(r_ref[i][j])
        new_ref.append(tmp)
      
    for i in range(len(r_ans)):
        an = r_ans[i]
        tmp = []
        for j in range(len(r_ans[i])):
            if r_ans[i][j]>on and r_ans[i][j] < off:
                tmp.append(r_ans[i][j])
        new_ans.append(tmp)        
        
    return new_ref, new_ans
    
def evaluate(r_ref, r_ans,fs =200, thr=30, ignore_sec = 0, length = 10, print_msg = True,):
   # print(length)
    if ignore_sec > 0:
        r_ref, r_ans = ignore_side(r_ref, r_ans,fs,length = length,ignore_sec = ignore_sec)
    all_TP = 0
    all_FN = 0
    all_FP = 0
    tol = int(thr*fs/1000)
    errors = []
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0
        detect_loc = 0
        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= tol)[0]
            detect_loc += len(loc)

            if len(loc) >= 1:
                
                TP += 1
                FP = FP + len(loc) - 1
                
                diff = r_ref[i][j] - r_ans[i][loc[0]]
                errors.append(diff/fs)
                
            elif len(loc) == 0:
                FN += 1
        FP = FP+(len(r_ans[i])-detect_loc)
        
        all_FP += FP
        all_FN += FN
        all_TP += TP
    if all_TP == 0:
        Recall = 0
        Precision = 0
        F1_score = 0
        Sen = 0
        PPV = 0
        
    else:
        Sen = all_TP / (all_FN + all_TP)
        PPV = 0
        Recall = all_TP / (all_FN + all_TP)
        Precision = all_TP / (all_FP + all_TP )
        F1_score = 2 * Recall * Precision / (Recall + Precision)
    if all_FP == 0:
        error_rate = 0
    else:
        error_rate =  all_FP / (all_FP + all_TP)
    if print_msg:
        print("TP's:{} FN's:{} FP's:{}".format(all_TP,all_FN,all_FP))
        print('Recall:{},Precision:{},F1-score:{}'.format(Recall,Precision, F1_score))
    
    return Recall,Precision, F1_score

def get_label(peaks,length,thr=50,fs = 1000, tol = 21):
    #tol = 17#21#21#21#int(thr*fs/1000)+1
    half = tol//2
    labels = np.zeros(length)
    for peak in peaks:
        if peak - half >= 0 and peak + half +1 < length:
            for i in range(peak-half, peak + half + 2):
                labels[i] = 1
    labels = labels.reshape((1, length))
    return labels

def gd(x, mu=0, sigma=1):
    """根据公式，由自变量x计算因变量的值
    Argument:
        x: array
            输入数据（自变量）
        mu: float
            均值
        sigma: float
            方差
    """
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu)**2 / (2 * sigma))
    return left * right






def spectral_rmse(signal1, signal2, fs=1.0, nperseg=256):
    """
    计算信号的光谱均方根误差（Spectral RMSE）。

    参数：
    - signal1: 第一个信号（一维数组或列表）
    - signal2: 第二个信号（一维数组或列表）
    - fs: 信号的采样率（默认为1.0）
    - nperseg: 每个段的长度，用于计算频谱（默认为256）

    返回：
    - spectral_rmse: 光谱均方根误差
    """
    # 计算信号的光谱
    _, _, Sxx1 = spectrogram(signal1, fs=fs, nperseg=nperseg)
    _, _, Sxx2 = spectrogram(signal2, fs=fs, nperseg=nperseg)

    # 计算频谱差异
    spectral_difference = np.abs(Sxx1 - Sxx2)

    # 计算均方根误差
    spectral_rmse = np.sqrt(mean_squared_error(Sxx1, Sxx2))

    return spectral_rmse
def spectral_correlation(signal1, signal2, fs=1.0, nperseg=256, mode='full'):
    """
    计算两个信号的光谱相关性。

    参数：
    - signal1: 第一个信号（一维数组或列表）
    - signal2: 第二个信号（一维数组或列表）
    - fs: 信号的采样率（默认为1.0）
    - nperseg: 每个段的长度，用于计算频谱（默认为256）
    - mode: 相关计算的模式（默认为'full'）

    返回：
    - spectral_corr: 光谱相关性
    """
    # 计算信号的光谱
    _, _, Sxx1 = spectrogram(signal1, fs=fs, nperseg=nperseg)
    _, _, Sxx2 = spectrogram(signal2, fs=fs, nperseg=nperseg)

    # 计算光谱的相关性
    spectral_corr = np.corrcoef(Sxx1.ravel(), Sxx2.ravel())[0, 1]
    return spectral_corr

from sklearn.metrics.pairwise import cosine_similarity


def COS_SIM(vector_a, vector_b):
    """
    Calculate the Cosine Similarity between two vectors.
    
    Parameters:
    vector_a (array-like): The first vector.
    vector_b (array-like): The second vector.
    
    Returns:
    float: The cosine similarity between the two vectors.
    """
    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)
    
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    return dot_product / (norm_a * norm_b)


def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred)) 

# def PRD(y, y_pred):
#     N = np.sum(np.square(y_pred - y))
#     D = np.sum(np.square(y_pred - np.mean(y)))

#     PRD = np.sqrt(N/D) * 100

#     return PRD



def PRD(original_signal, reconstructed_signal):
    """
    Calculate the Percentage Root Mean Square Difference (PRD) between two signals.
    
    Parameters:
    original_signal (array-like): The original signal.
    reconstructed_signal (array-like): The reconstructed signal.
    
    Returns:
    float: The PRD value in percentage.
    """
    original_signal = np.asarray(original_signal)
    reconstructed_signal = np.asarray(reconstructed_signal)
    
    numerator = np.sum((original_signal - reconstructed_signal) ** 2)
    denominator = np.sum(original_signal ** 2)
    
    prd = np.sqrt(numerator / denominator) * 100
    return prd


def SNR(original_signal, estimated_signal):
    # 计算原始信号的能量
    signal_power = np.sum(np.abs(original_signal)**2)
    
    # 计算误差信号的能量
    error_power = np.sum(np.abs(original_signal - estimated_signal)**2)
    
    # 计算信噪比 (以分贝为单位)
    snr_db = 10 * np.log10(signal_power / error_power)
    return snr_db

def RRMSE(y, y_pred):
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # 计算实际观测值范围
    range_actual = np.max(y) - np.min(y)

    # 计算相对均方根误差（RRMSE）
    rrmse = (rmse / range_actual) * 100
    return rrmse

def evaluate_signal(ground_, pred_,fs=200,print_msg = False):
    
    ground =scale(ground_)#*0.5
    pred = scale(pred_)#*0.5
    
    pcc, _ = stats.pearsonr(ground, pred)
    pcc = abs(pcc)
    snr = SNR(ground, pred)
   # csim = COS_SIM(ground, pred)
    prd = PRD(ground, pred)
    rrmse = RRMSE(ground, pred)
    ssd = SSD(ground, pred)
    spec_corr = spectral_correlation(ground, pred,fs = fs)
   # spec_rmse = spectral_rmse(ground, pred,fs = fs)
    mae = np.mean(np.abs(ground-pred))
    mse = np.mean(np.power(ground-pred, 2))
    if print_msg:
        print('pcc:',pcc)
    return pcc,snr,prd,spec_corr, mae, mse#ssd,csim,mse#ssd,rrmse


def sild_window(locs,fhrs,fs):
    record_minute = 5
    seconds = record_minute*60
    L = seconds*fs
    #print(L)
    w_L =5*fs #5*fs
    shift = fs#fs
    m_fhrs = []
    for i in range(0, L - w_L , shift):
        on,off = i, i+w_L
        w_fhs = []
        for j in range(len(locs)):
            if locs[j] >= on and  locs[j] < off:
                w_fhs.append(fhrs[j])
        if len(w_fhs) == 0:
            m_fhrs.append(0)
      #      print(i/(fs*60))
        else:
            m_fhrs.append(np.median(w_fhs))
    return m_fhrs    
                
    
def get_fhr(peaks, fs, window_size= 30):
    fhrs =60 * fs/ np.diff(peaks)
    locs = []
    for i in range(len(peaks)-1):
        locs.append((peaks[i]+peaks[i+1])//2)
    
    m_fhrs = sild_window(locs,fhrs,fs)
    return m_fhrs


def get_mean_fhr(peaks, fs):
    fhrs =60 * fs/ np.diff(peaks)
    cnt = len(fhrs)
    tmp = int(0.1*cnt)
    return np.mean(sorted(fhrs)[tmp:-tmp])

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
    
    
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    
    
def speed_up():

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + 1 - opt.epochs/2) / float(opt.epochs/2 + 1)
            lr_l = (1 - epoch / opt.epochs) ** 0.9
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)#, threshold=0.01
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


