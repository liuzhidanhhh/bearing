# 旋转机械故障检测工程文件说明

### 文件夹

**mat-data**：存放从[Case Western Reserve university](http://csegroups.case.edu/bearingdatacenter/pages/12k-drive-end-bearing-fault-data) 下载的数据，实验中所用的是故障直径为0.007‘’ 的四种转速的normal、inner、ball、outer 6:00的故障数据。

**512-data：** 存放从原始信号中截取的长度为512个连续采样信号数据集，命名按照：转速_数据类型存放，例如1730_train.npy 或者1730_label.npy ，为python 存取处理方便所有数据文件均用npy 格式存储。

**512-scale-data**： 归一化后的512-data 数据集。

**scale-gen-512-data**： 生成的信号长度为512的数据，命名规则：转速_gen_data0/1/2/3.npy

data0:生成的normal数据，data1，生成的inner数据，以此类推。

**checkpoint_dir** : 存放cnn 训练后的模型，避免测试时重复训练。

**scale_checkpoint_dir:** 存放归一化后的数据用cnn 训练后的模型。

**model_data**: 存放验证WGAN 有效性模型架构的数据 

格式：normal:(inner+gen_inner):(ball+gen_ball):(outer+gen_outer)

data1:180:180:180:180

data2: 450:10:10:10； 

 data3:450:5:5:5   	

data4:180:(90+90):(90+90):(90+90)

data5:360:(90+30):(90+30):(90+30)

data6: 450:(0+90):(0+90):(0+90)

### 代码

#### 非稳态故障检测主要代码

**genDataset.py** ： 利用原始数据集生成我们需要的数据并存放在512-data中

**scale.py ** : 归一化512-data 中的数据，并存入512-scale-data

**calculate_ES.py ** : 计算原始的数据包络谱

**convolution.py ** : ES+CNN 架构代码，用于解决非稳态下故障检测

#### 样本不均衡问题主要代码

**read_data_set.py** :  用于从处理好的数据集中读取wgan所需的故障类型的代码

**wgan.py **: 生成outer 类型所需的wgan网络

**wgan1.py **: 生成inner类型的WGAN网络

**wgan2.py **: 生成ball 类型的wgan网络

**test1.py**: 将WGAN 生成的网络放进前面训练好的ES+cnn 模型中进行测试，以判断生成信号的好坏

#### 验证GAN 生成数据做信号增强确实能够解决样本不均衡问题的代码

**gen_model_data.py** ： 生成数据 用于验证GAN做数据增强能够解决样本极度不均衡场景下故障检测问题

**conv_model.py** : 用验证 gen_model_data的组合数据 的主要模型代码

**model_test.py **: 调用conv_model.py 用不同数据集生成不同数据组合下的结果

#### 其他

**draw.py ** 原始数据集挖掘数据特征用的代码，主要是画原始数据的时域、频域特征

**draw_gen.py ** 画用于训练的归一化或者不归一化的数据、GAN生成的数据的的时域特征、频域特征

**extract_feature.py **： 提取信号均值、峰值、ppm等特征值的函数

**plot_gen_scater.py**: 画生成数据集的三维散点图的函数

**plot_real_scater.py **: 画原数据的散点图

