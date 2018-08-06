# 画生成数据集的散点图

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from extract_feature import timeDomainFeature
from calculate_ES import cal_es
fig = plt.figure()
ax = Axes3D(fig)
for i in range(4):
    data=np.load('1730_gen_data'+str(i)+'.npy')
    data=data[5:9]
    data=np.array(data)
    new=np.reshape(data,[np.shape(data)[0]*np.shape(data)[1],np.shape(data)[2]])
    new_data=[cal_es(x,128) for x in new]
    feature = [timeDomainFeature (x) for x in new_data]
    feature=np.array(feature)
    ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2])
plt.show()