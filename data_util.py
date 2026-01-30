import pyedflib
from scipy.signal import butter, filtfilt,sosfiltfilt
from config import cfg
import wfdb
import numpy as np
import scipy.signal
from wfdb.io import rdrecord
import file_util as fu
from sklearn.preprocessing import scale
import scipy
from sklearn.preprocessing import MinMaxScaler

def get_dblist():
    return ['addb']

def get_namelist(db):
    func = {'addb':get_namelist_from_addb}
    return func[db]()
    
def get_data(db, name, channel,return_fecg):
    func = {'addb':get_data_from_addb}
    return func[db](name,channel,return_fecg)

def get_db_msg(db):
    msg = {'addb':{'datapath': cfg().ADFECGDB, 'fs':1000,'contain_fecg':True}, 
          }
    db_msg = msg[db]
    
    return db_msg['datapath'], db_msg['contain_fecg'], db_msg['fs'] 






def get_data_from_addb(name, channel = [0],return_fecg=False):
    path = cfg().ADFECGDB + '/'
    fs = 1000
    f = pyedflib.EdfReader(path + name)
    ecg = []
    for i in range(len(channel)):
        ecg_tmp = f.readSignal(channel[i]+1)
        ecg.append(ecg_tmp)
    ecg = np.array(ecg)
    ecg = ecg.reshape((len(channel),len(ecg[0])))

    signal_annotation = wfdb.rdann(path + name, "qrs", sampfrom=0, sampto=60000*5)

    peaks = signal_annotation.sample

    if return_fecg:
        fecg = f.readSignal(0)
        fecg = fecg.reshape((1,len(fecg)))

        return ecg,fecg, peaks
    
    return ecg, peaks



    
def remove_nan(arr):  # 去除信号中的无效值
    for _ in range(3):
        nan_flag = np.isnan(arr)
        for i in range(len(nan_flag)):
            if nan_flag[i] == True and i <= (len(arr) / 2):
                for j in range(1, 20):
                    if nan_flag[i + j] == False:
                        arr[i] = arr[i + j]
                        break
            if nan_flag[i] == True and i > (len(arr) / 2):
                for j in range(1, 20):
                    if nan_flag[i - j] == False:
                        arr[i] = arr[i - j]
                        break
    return arr


def pad_audio(audio, segment_samples):
    r"""Pad the audio with zero in the end so that the length of audio can
    be evenly divided by segment_samples.

    Args:
        audio: (channels_num, audio_samples)

    Returns:
        padded_audio: (channels_num, audio_samples)
    """
    channels_num, audio_samples = audio.shape

    # Number of segments
    segments_num = int(np.ceil(audio_samples / segment_samples))

    pad_samples = segments_num * segment_samples - audio_samples

    padded_audio = np.concatenate(
        (audio, np.zeros((channels_num, pad_samples))), axis=1
    )
    # (channels_num, padded_audio_samples)

    return padded_audio

def enframe(sig, segment_samples):
    audio_samples = sig.shape[1]
    hop_samples =  segment_samples // 2
    
    segments = []
    
    pointer = 0
    while pointer + segment_samples <= audio_samples:
        segments.append(sig[:, pointer : pointer + segment_samples])
        pointer += hop_samples

    segments = np.array(segments)
    
    return segments

def deframe(segments):
    def _is_integer(x: float) -> bool:
        if x - int(x) < 1e-10:
            return True
        else:
            return False

    
    (segments_num, _, segment_samples) = segments.shape

    if segments_num == 1:
        return segments[0]

    assert _is_integer(segment_samples * 0.25)
    assert _is_integer(segment_samples * 0.75)

    output = []

    output.append(segments[0, :, 0 : int(segment_samples * 0.75)])

    for i in range(1, segments_num - 1):
        output.append(
            segments[
                i, :, int(segment_samples * 0.25) : int(segment_samples * 0.75)
            ]
        )

    output.append(segments[-1, :, int(segment_samples * 0.25) :])

    output = np.concatenate(output, axis=-1)
    output = output.flatten()
    return output

def __butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, axis=1,filter_type = 'band'):
    if filter_type == 'band':
        b, a = __butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
    elif filter_type == 'sos':
        y = sos_filter(data, lowcut, highcut, fs, order,axis)
    return y

def sos_filter(signal, lowcut, highcut, fs = 20, order=4,axis = 0):
    """The parameter taken in here is the Poly5 file. Output is
    the EMG after a bandpass as made here.

    :param data_emg: Poly5 file with the samples to work over
    :type data_emg: ~TMSiSDK.file_readers.Poly5Reader
    :param low_pass: The number to cut off :code:`frequenciesabove`
    :type low_pass: int
    :param high_pass: The number to cut off :code:`frequenceisbelow`
    :type high_pass: int

    :returns: The bandpass filtered EMG sample data
    :rtype: ~numpy.ndarray
    """
    signal_ = signal.copy()
    sos = butter(
        order,
        [lowcut, highcut],
        'bandpass',
        fs=fs,
        output='sos',
    )
    # sos (output parameter) is second order section  -> "stabilizes" ?
    signal_ = sosfiltfilt(sos, signal_, axis=axis)
    return signal_#np.asarray(y).flatten()




def iirnotch(data, fs):
    w3, w4 = scipy.signal.iirnotch(50, 50, fs=fs)  # 50hz陷波滤波器
    w5, w6 = scipy.signal.iirnotch(60, 50, fs=fs)  # 60hz陷波滤波器
    
    data = scipy.signal.filtfilt(w3, w4, data, axis = 1)
    data = scipy.signal.filtfilt(w5, w6, data, axis = 1)
    return data

def preprocess(signal, filter, origin_fs, target_fs):
    if filter is not None:
        signal = butter_bandpass_filter(signal,filter[0], filter[1], origin_fs, axis = 1)

   # signal = iirnotch(signal, origin_fs)

    signal = iirnotch(signal, origin_fs)
    signal = scale(signal, axis=1)
    scale_fs = origin_fs/target_fs
  
    signal = scipy.signal.resample(signal, int(signal.shape[1] / scale_fs), axis=1)

    return signal


def get_namelist_from_addb():
    namelist =  [ "r01.edf", "r04.edf", "r07.edf","r08.edf", "r10.edf"]
    return namelist



