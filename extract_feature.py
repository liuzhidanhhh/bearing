# 求各种信号特征的函数

import numpy as np
from scipy.fftpack import fft,ifft



#1 求平均
def mean(arr):
    return sum(arr)/len(arr)

#2 计算
def variance(arr):
    t1=mean(arr)
    sum=0
    for stu in arr:
        sum+=pow(stu-t1,2)
    if len(arr)-1 == 0:
        return 0
    return pow(sum/len(arr)-1,0.5)

#3 计算均方根
def rootmeansquare(arr):
    t1=mean(arr)
    return pow(t1,0.5)

#4 计算峰值
def peak(arr):
    sum=0
    for stu in arr:
        if abs(stu)>sum:
            sum=abs(stu)
    return sum

#5
def skewness(arr):
    t1=mean(arr)
    t2=variance(arr)
    sum=0
    for stu in arr:
        sum+=pow(stu-t1,3)
    if (len(arr)-1)*pow(t2,3)==0:
        return 0
    return sum/(len(arr)-1)*pow(t2,3)

#6
def kurtosis(arr):
    t1=mean(arr)
    t2=variance(arr)
    sum=0
    for stu in arr:
        sum+=pow(stu-t1,4)
    if (len(arr)-1)*pow(t2,4)==0:
        return 0
    return sum/(len(arr)-1)*pow(t2,4)

#7
def crestFactor(arr):
    t3=rootmeansquare(arr)
    sum=0
    for stu in arr:
        sum+=abs(stu)
    if sum/len(arr) == 0:
        return 0
    return t3/(sum/len(arr))

#8
def pulseindicator(arr):
    t4=peak(arr)
    sum=0
    for stu in arr:
        sum+=pow(abs(stu),0.5)
    if pow(sum/len(arr),2)==0:
        return 0
    return t4/pow(sum/len(arr),2)

#9
def shapeFactor(arr):
    t4=peak(arr)
    t3=rootmeansquare(arr)
    if t3==0:
        return 0
    return t4/t3

#10
def latitudefactor(arr):
    t4=peak(arr)
    sum=0
    for stu in arr:
        sum+=abs(stu)
    if sum==0:
        return 0
    return t4/(sum/len(arr))

#11
def maxGradient(arr):
    return np.max(np.gradient(arr))

#12
def ppm(arr):
    nfft = 128
    p = abs (fft (arr, nfft))
    return max(p)/mean(p)
#12
def ppmVar(arr):
    nfft=128
    p=abs(fft(arr,nfft))
    return np.var(p)

def timeDomainFeature(arr):
    return [maxGradient(arr),ppm(arr),ppmVar(arr)]


