import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft

def cal_es(signal,window_size):
    ''' 返回window_size 大小的包络谱信号'''

    # 计算hilbert 信号值
    BA=hilbert(signal)
    hb_signal=np.abs(BA)

    # 计算傅立叶信号值
    fft_signal=fft(hb_signal,window_size)
    es=np.abs(fft_signal)/len(fft_signal)

    return es