#画真实数据的散点图

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from extract_feature import timeDomainFeature

data=np.load('pre-data/1730_train.npy')
label=np.load('pre-data/1730_label.npy')
train0=[]
train1=[]
train2=[]
train3=[]
label0 = []
label1 = []
label2 = []
label3 = []
for i in range (len (data)):
    if i % 4 == 0:
        train0.append (data[i])
        label0.append (label[i])
    elif i % 4 == 1:
        train1.append (data[i])
        label1.append (label[i])
    elif i % 4 == 2:
        train2.append (data[i])
        label2.append (label[i])
    else:
        train3.append (data[i])
        label3.append (label[i])

feature=[timeDomainFeature(x) for x in train0]
feature1=[timeDomainFeature(x) for x in train1]
feature2=[timeDomainFeature(x) for x in train2]
feature3=[timeDomainFeature(x) for x in train3]
feature=np.array(feature)
feature1=np.array(feature1)
feature2=np.array(feature2)
feature3=np.array(feature3)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(feature[:,0], feature[:,1], feature[:,2],label='normal')
ax.scatter(feature1[:,0], feature1[:,1], feature1[:,2],label='inner')
#ax.scatter(feature2[:,0], feature2[:,1], feature2[:,2],label='ball')
#ax.scatter(feature3[:,0], feature3[:,1], feature3[:,2],label='outer')
plt.show()