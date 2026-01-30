import pyedflib
from scipy.signal import butter, filtfilt
from config import cfg
import wfdb
import numpy as np
import scipy.signal
from wfdb.io import rdrecord
import file_util as fu
from sklearn.preprocessing import scale
import scipy
import numpy as np
import pywt
from scipy.signal import firwin2, filtfilt, savgol_filter
import math



def get_max_abs(sig):  # 将信号放缩到-1到1
    max_abs = max(abs(max(sig)), abs(min(sig)))
    return sig / max_abs
    

def fir_filt(window, fs=500):
    """
    Remove the baseline ("flattening the segment")
    and powerline noise of segment.
    window ~ signal segment - float array
    """

    if len(window) > 93:
        # Minimum length for filter is 94 to satisfy
        # window.shape[-1] > padlen = 3 * numtaps
        gain = [0, 1, 1, 0, 0]
        freq = [0, 1, 45, 55, fs / 2]

        b = firwin2(31, freq, gain, fs=fs, window='hamming', antisymmetric=True)
        window = filtfilt(b, 1, window)

    return window


def interp_nan(window):
    """
    Interpolate NaNs after outlier removal.
    window  : 3 sec segment ~ Array float
    """
    window = np.array(window)
    isnan = np.isnan(window)
    nans = np.where(isnan)[0]

    # No NaNs; No interpolation
    if np.size(nans) == 0:
        return window

    # If the whole signal is NaN set signal to zero
    elif len(nans) == len(window):
        window[:] = 0

    # Fill NaNs using linear interpolation (NaNs on edge are copies of outermost float)
    else:
        ok = ~isnan
        xp = ok.ravel().nonzero()[0]
        fp = window[ok]
        x = isnan.ravel().nonzero()[0]
        window[isnan] = np.interp(x, xp, fp)

    return window

def __butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, axis=1):
    b, a = __butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def iirnotch(data, fs):
    w3, w4 = scipy.signal.iirnotch(50, 50, fs=fs)  # 50hz陷波滤波器
    w5, w6 = scipy.signal.iirnotch(60, 50, fs=fs)  # 60hz陷波滤波器
    
    data = scipy.signal.filtfilt(w3, w4, data, axis = 1)
    data = scipy.signal.filtfilt(w5, w6, data, axis = 1)
    return data

def preprocess(signal, filter, origin_fs, target_fs):
    signal = butter_bandpass_filter(signal,filter[0], filter[1], origin_fs, axis = 1)

   # signal = iirnotch(signal, origin_fs)

    signal = iirnotch(signal, origin_fs)
    signal = scale(signal, axis=1)
    scale_fs = origin_fs/target_fs
  
    signal = scipy.signal.resample(signal, int(signal.shape[1] / scale_fs), axis=1)

    return signal