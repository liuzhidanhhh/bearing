import tensorflow as tf
import numpy as np
import calculate_ES
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



NUM_LABELS = 4
VALIDATION_SIZE = 32  # Size of the validation set.
SEED =15 # Set to None for random seed.
BATCH_SIZE = 16
NUM_EPOCHS = 85
EVAL_BATCH_SIZE = 32
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
window_size=128
Depth = 4
Window_Length=128
Window_Wide=1
Window_Depth=1

point=0
num=32
def draw_real():
    # 绘制真实数据的 cnn散点图
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph ('./scale-checkpoint_dir/MyModel-3500.meta')
        saver.restore (sess, tf.train.latest_checkpoint ('./scale-checkpoint_dir'))
        graph = tf.get_default_graph ()
        eval_data = graph.get_tensor_by_name ('eval_data:0')
        feature = graph.get_tensor_by_name ("feature_1:0")

        test_data = np.load ('512-scale-data/1797_train.npy')
        test_labels1 = np.load ('512-scale-data/1797_label.npy')
        test_data1 = [calculate_ES.cal_es (x, window_size) for x in test_data]
        test_data1 = np.reshape (test_data1, [len (test_data1), 128, 1, 1])
        res=[]
        for i in range(11):
            test=test_data1[32*i:32*(i+1)]
            fea=sess.run(feature,feed_dict={eval_data:test})
            res.append(fea)
        res=np.array(res).reshape([352,64])

        pca = PCA (n_components=3)
        newData = pca.fit_transform (res)
        np.save('newData.npy',newData)

        normal = []
        inner = []
        ball = []
        outer = []

        for i in range(87):
            normal.append(list(newData[4*i]))
            inner.append(list(newData[4*i+1]))
            ball.append(list(newData[4*i+2]))
            outer.append(list(newData[4*i+3]))
        normal=np.array(normal)
        np.save('normal.npy',normal)
        inner=np.array(inner)
        ball=np.array(ball)
        outer=np.array(outer)

        fig = plt.figure ()
        ax = Axes3D (fig)
        normal=ax.scatter(normal[:,0],normal[:,1],normal[:,2])
        inner=ax.scatter(inner[:,0],ball[:,1],ball[:,2])
        ball=ax.scatter(ball[:,0],ball[:,1],ball[:,2])
        outer=ax.scatter(outer[:,0],outer[:,1],outer[:,2])
        ax.legend((normal,inner,ball,outer),('normal','inner','ball','outer'))
        plt.show()

def draw_gen():
    # 绘制生成数据的cnn 特征散点图
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph ('./scale-checkpoint_dir/MyModel-3500.meta')
        saver.restore (sess, tf.train.latest_checkpoint ('./scale-checkpoint_dir'))
        graph = tf.get_default_graph ()
        eval_data = graph.get_tensor_by_name ('eval_data:0')
        feature = graph.get_tensor_by_name ("feature_1:0")

        inner = np.load ('scale-gen-512-data/1730_gen_data1.npy')[1]
        ball = np.load ('scale-gen-512-data/1730_gen_data2.npy')[-1]
        outer = np.load ('scale-gen-512-data/1730_gen_data3.npy')[-1]

        test_data=np.concatenate((inner,ball,outer),axis=0)

        test_data1 = [calculate_ES.cal_es (x, window_size) for x in test_data]
        test_data1 = np.reshape (test_data1, [len (test_data1), 128, 1, 1])
        res=[]
        for i in range(3):
            test=test_data1[32*i:32*(i+1)]
            fea=sess.run(feature,feed_dict={eval_data:test})
            res.append(fea)
        res=np.array(res).reshape([96,64])

        pca = PCA (n_components=3)
        newData = pca.fit_transform (res)
        normal=np.load('normal.npy')[:32]
        #normal = newData[:32]
        inner = newData[0:32]
        ball = newData[32:64]
        outer = newData[64:96]


        fig = plt.figure ()
        ax = Axes3D (fig)
        normal=ax.scatter(normal[:,0],normal[:,1],normal[:,2])
        inner=ax.scatter(inner[:,0],ball[:,1],ball[:,2])
        ball=ax.scatter(ball[:,0],ball[:,1],ball[:,2])
        outer=ax.scatter(outer[:,0],outer[:,1],outer[:,2])
        ax.legend((normal,inner,ball,outer),('normal','inner','ball','outer'))
        plt.show()
draw_real()
draw_gen()