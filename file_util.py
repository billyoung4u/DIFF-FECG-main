import pickle
import pandas as pd
import os
from datetime import datetime

def save(variable, filename):
    #保存python变量
    c_file = open(filename,'wb')
    pickle.dump(variable, c_file,protocol = 4)
    c_file.close()
    
def load(filename):
    d_file = open(filename,'rb+')
    data = pickle.load(d_file)
    d_file.close()
    return data

def save_csv(result,filename,columns=None):
    df = pd.DataFrame(result,columns=columns)
    df.to_csv(filename, index = False)
    print('save to csv:',filename, '...success...')

def exists(file):
    return os.path.exists(file)

def create_dir(path):
    if exists(path) is not True:
        os.mkdir(path)
        
        
def get_time_str():
    now = datetime.now()
    date_string = now.strftime("%m-%d_%H-%M")
  #  print("当前时间为：", date_string)
    return date_string