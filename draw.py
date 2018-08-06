# 初步画各信号包络谱

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fftpack import fft

window_size=128
x=np.linspace(1,128,128)
normal=pd.read_csv('csv-data/1797/normal.csv',header=0)
inner=pd.read_csv('csv-data/1797/inner.csv',header=0)
ball=pd.read_csv('csv-data/1797/ball.csv',header=0)
outer=pd.read_csv('csv-data/1797/outer_6.csv',header=0)

normal.columns=['indexx','DE','FE']
inner.columns=['indexx','BA','DE','FE']
ball.columns=['indexx','BA','DE','FE']
outer.columns=['indexx','BA','DE','FE']


def cal_es(signal):
    BA=hilbert(signal)
    hb_signal=np.abs(BA)
    fft_signal=fft(hb_signal,window_size)
    es=np.abs(fft_signal)/len(fft_signal)
    return es

nor_sig=cal_es(normal.DE)
inn_sig=cal_es(inner.DE)
ball_sig=cal_es(ball.DE)
out_sig=cal_es(outer.DE)

fig=plt.figure()
ax0=fig.add_subplot(411)
ax0.plot(x[1:],nor_sig[1:])
ax0.set_xlabel('Frequency[Hz]')
ax0.set_ylabel('Amplitude')
ax1=fig.add_subplot(412)
ax1.plot(x[1:],inn_sig[1:])
ax1.set_xlabel('Frequency[Hz]')
ax1.set_ylabel('Amplitude')
ax2=fig.add_subplot(413)
ax2.plot(x[1:],ball_sig[1:])
ax2.set_xlabel('Frequency[Hz]')
ax2.set_ylabel('Amplitude')
ax3=fig.add_subplot(414)
ax3.plot(x[1:],out_sig[1:])
ax3.set_xlabel('Frequency[Hz]')
ax3.set_ylabel('Amplitude')
fig.suptitle('Envelope Spectrums(DE)')
