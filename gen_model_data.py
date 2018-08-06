# 生成 用于构建 模型 测试 GAN 扩增数据集的有效性 的数据集
# 生成的数据存储在model_data 中


import numpy as np


def draw_real_data(data_path, label_path, tag, num):
    ''' 从真实数据集中抽取取label为tag的num 个数据'''
    data = np.load (data_path)
    label = np.load (label_path)
    res_data = []
    res_label = []
    for i in range (len (data)):
        if i % 4 == tag and i < 4 * num:
            res_data.append (data[i])
            res_label.append (label[i])

    return np.array(res_data), np.array(res_label)


def draw_gen_data(data_path,label_path,num):
    ''' 从真实数据集中抽取出 num 个数据 '''
    if num!=0:
        data = np.load (data_path)
        data = data.reshape (data.shape[0] * data.shape[1], data.shape[2])
        labels = np.load (label_path)
        label = np.reshape (labels, [labels.shape[0] * labels.shape[1]])

        res_data = data[-(num + 1):-1]
        res_label = label[-(num + 1):-1]

        return res_data, res_label
    else:
        return [],[]

def make_data_set(real_data_path, real_label_path, real_num1, real_num2, real_num3, real_num4,
                gen_data_path, gen_label_path, gen_num1, gen_num2, gen_num3):
    ''' 生成混合数据集 '''

    normal, nor_label = draw_real_data (real_data_path, real_label_path, 0, real_num1)
    inner, inn_label = draw_real_data (real_data_path, real_label_path, 1, real_num2)
    ball, ball_label = draw_real_data (real_data_path, real_label_path, 2, real_num3)
    outer, out_label = draw_real_data (real_data_path, real_label_path, 3, real_num4)

    gen_inner, gen_inn_label = draw_gen_data (gen_data_path+'1.npy', gen_label_path+'1.npy', gen_num1)
    gen_ball, gen_ball_label = draw_gen_data (gen_data_path+'2.npy', gen_label_path+'2.npy', gen_num2)
    gen_outer, gen_out_label = draw_gen_data (gen_data_path+'3.npy', gen_label_path+'3.npy', gen_num3)


    # 合并数据集
    if gen_num1==0:
        res_data=np.concatenate((normal,inner,ball,outer),axis=0)
        res_label=np.concatenate((nor_label,inn_label,ball_label,out_label),axis=0)
    elif real_num2==0:
        res_data=np.concatenate((normal,gen_inner,gen_ball,gen_outer),axis=0)
        res_label=np.concatenate((nor_label,gen_inn_label,gen_ball_label,gen_out_label),axis=0)
    else:
        res_data = np.concatenate((normal, inner,ball,outer,gen_inner,gen_ball,gen_outer),axis=0)
        res_label=np.concatenate((nor_label,inn_label,ball_label,out_label,gen_inn_label,gen_ball_label,gen_out_label),axis=0)


    # 打乱数据集
    np.random.seed(10)
    np.random.shuffle(res_data)
    np.random.seed(10)
    np.random.shuffle(res_label)

    return res_data, res_label

# save data
real_data_path='512-scale-data-450/1730_train.npy'
real_label_path='512-scale-data-450/1730_label.npy'
gen_data_path='scale-gen-512-data/1730_gen_data'
gen_label_path='scale-gen-512-data/1730_gen_label'
# save data1
real_num1=450
real_num2=0
real_num3=0
real_num4=0
gen_num1=10
gen_num2=10
gen_num3=10
data,label=make_data_set(real_data_path,real_label_path,real_num1,real_num2,real_num3,real_num4,
                  gen_data_path,gen_label_path,gen_num1,gen_num2,gen_num3)
np.save('model_data/data6.npy',data)
np.save('model_data/label6.npy',label)




