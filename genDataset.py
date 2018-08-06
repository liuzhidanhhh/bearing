# script to generate samples saved with npy format
# each dataset having 360 samples
# Data with same rpm is grouped as one dataset which comprises four bearing condtions with 90 samples for each condition

# Data come from http://csegroups.case.edu/bearingdatacenter/pages/download-data-file
# rotational speed：1797rpm and 1730rmp
# frequency: 12KHZ
# 从给定的数据集中随机截取size个样本点 截num_size个样本每种条件

# 取driven end 的数据
#label normal,inner,ball,outer:0,1,2,3


import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft
import re
import calculate_ES


path = 'mat-data'
sub_path = ['1797','1730','1772','1750']
file_name=['normal','inner','ball','outer_6']
window_size=128
size=512
num_sample=450


for i in range(len(sub_path)):
    sig=[]
    for j in range(len(file_name)):
        features_struct = io.loadmat('mat-data/7/'+ sub_path[i]+'/'+file_name[j]+'.mat')
        keys = list (features_struct.keys ())
        name=[]
        for x in keys:
            name.append(re.findall(r'[X0-9]*_DE_[a-z]*',x))
        for x in name:
            if x!=[]:
                De=x[0]
        DE = pd.DataFrame(features_struct[De])
        DE.columns=['DE']
        De=np.array (DE).reshape (len (DE))
        sig.append(De)

    dataset=[]
    label=[]
    for j in range(num_sample):
        np.random.seed(j)
        x=np.random.randint(12000)
        dataset.append(sig[0][x:x+size])
        label.append(0)
        dataset.append(sig[1][x:x+size])
        label.append(1)
        dataset.append (sig[2][x:x + size])
        label.append (2)
        dataset.append (sig[3][x:x + size])
        label.append (3)
    np.save('512-data-450/'+sub_path[i]+'_train.npy',dataset)
    np.save('512-data-450/'+sub_path[i] + '_label.npy',label)

    '''
    es_data = []
    for x in dataset:
        es_data.append(calculate_ES.cal_es(x,window_size))
    np.save('pre-data/'+sub_path[i]+'_train.npy',es_data)
    np.save('pre-data/'+sub_path[i]+'_label.npy',label)
    '''