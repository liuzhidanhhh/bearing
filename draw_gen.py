#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

path='512-scale-data/1730_train.npy' #数据是混合在一起的  所以需要分开
data=np.load(path)
label=np.load('512-scale-data/1730_label.npy')

def time(data,begin,end,ylim):
    # data: 混合数据集0、1、2、3（4，5，6，7...）分别为normal、inner、ball、outer对应的数据
    # begin: 从数据集中的begin位置出取数据,begin需要是4的倍数，保证绘制的第一张图是normal
    # end：绘制从0-end 的长度

    fig=plt.figure()
    x=range(end)
    ax0=fig.add_subplot(411)
    ax0.plot(x[1:],data[begin][1:end],label='normal')
    ax0.set_xlabel('time')
    ax0.set_ylabel('normal')
    ax0.set_ylim(-ylim,ylim)
    ax1=fig.add_subplot(412)
    ax1.plot(x[1:],data[begin+1][1:end])
    ax1.set_xlabel('time')
    ax1.set_ylabel('inner')
    ax1.set_ylim(-ylim,ylim)
    ax2=fig.add_subplot(413)
    ax2.plot(x[1:],data[begin+2][1:end])
    ax2.set_xlabel('time')
    ax2.set_ylabel('ball')
    ax2.set_ylim(-ylim,ylim)
    ax3=fig.add_subplot(414)
    ax3.plot(x[1:],data[begin+3][1:end])
    ax3.set_xlabel('time')
    ax3.set_ylabel('outer')
    ax3.set_ylim(-ylim,ylim)

    fig.suptitle('Amplitude-time')

time(data,200,512,1)


def time_gen(data,pos,end,ylim):
    # data: 数据名字列表
    # begin: 从数据集中的begin位置出取数据,begin需要是4的倍数，保证绘制的第一张图是normal
    # end：绘制从0-end 的长度
    normal=np.load(data[0])[pos]
    inner=np.load(data[1])[pos]
    ball=np.load(data[2])[pos]
    outer=np.load(data[3])[pos]

    fig = plt.figure ()
    x = range (end)
    ax0 = fig.add_subplot (411)
    ax0.plot (x[1:],normal[-1][1:end], label='normal')
    ax0.set_xlabel ('time')
    ax0.set_ylabel ('normal')
    ax0.set_ylim (-ylim, ylim)
    ax1 = fig.add_subplot (412)
    ax1.plot (x[1:], inner[-1][1:end])
    ax1.set_xlabel ('time')
    ax1.set_ylabel ('inner')
    ax1.set_ylim (-ylim, ylim)
    ax2 = fig.add_subplot (413)
    ax2.plot (x[1:], ball[-1][1:end])
    ax2.set_xlabel ('time')
    ax2.set_ylabel ('ball')
    ax2.set_ylim (-ylim, ylim)
    ax3 = fig.add_subplot (414)
    ax3.plot (x[1:], outer[-1][1:end])
    ax3.set_xlabel ('time')
    ax3.set_ylabel ('outer')
    ax3.set_ylim (-ylim, ylim)

    fig.suptitle ('generated data after scaled')

gen_data=['gen-512-data/1730_gen_data0.npy','gen-512-data/1730_gen_data1.npy','gen-512-data/1730_gen_data2.npy','gen-512-data/1730_gen_data3.npy']
time_gen(gen_data,10,512,2)
