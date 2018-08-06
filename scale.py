# 归一化数据集

import numpy as np

def scale(dataset):
    max=np.max(dataset)
    min=np.min(dataset)
    res=[x/(max-min) for x in dataset]
    return res

train_path='512-data-450/1730_train.npy'
label_path='512-data-450/1730_label.npy'

train=np.load(train_path)
label=np.load(label_path)

train=scale(train)

np.save('512-scale-data-450/1730_train.npy',train)
np.save('512-scale-data-450/1730_label.npy',label)