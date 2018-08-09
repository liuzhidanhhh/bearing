# ES + cnn 架构代码，用于4种选装机械状态（normal、inner、ball、outer）分类

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import calculate_ES

NUM_LABELS = 4
VALIDATION_SIZE = 32  # 验证集的大小
SEED =15
BATCH_SIZE = 16
NUM_EPOCHS = 85
EVAL_BATCH_SIZE = 32
EVAL_FREQUENCY = 100  # 评估的频率，输出测试结果的频率
window_size=128       # 计算ES 的窗口大小
Depth = 4
Window_Length=128
Window_Wide=1
Window_Depth=1

path='512-scale-data/'
train_dataset='1730'
test_dataset=['1797','1750','1772']
point=0


def error_rate(predictions, labels):
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])


def main(_):

    # 加载训练集和测试集 并 计算包络普信号
    train_data = np.load(path+train_dataset+'_train.npy')
    train_data=[calculate_ES.cal_es(x,window_size) for x in train_data]
    train_labels = np.load(path+train_dataset+'_label.npy')
    test_data = np.load(path+test_dataset[point]+'_train.npy')
    test_data=[calculate_ES.cal_es(x,window_size) for x in test_data]
    test_labels = np.load(path+test_dataset[point]+'_label.npy')
    train_data=np.reshape(train_data,[len(train_data),128,1,1])
    test_data=np.reshape(test_data,[len(test_data),128,1,1])

    #划分验证集
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]

    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # 训练 测设节点是大小设置
    train_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, Window_Length, Window_Wide, Window_Depth))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder (tf.float32, shape=(EVAL_BATCH_SIZE, Window_Length, Window_Wide, Window_Depth),name='eval_data')

    # 卷积层网络设置
    conv1_weights = tf.Variable(tf.truncated_normal([5, 1, Window_Depth, Depth], stddev=0.1, seed=SEED, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([4], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.truncated_normal([5, 1, Depth, Depth], stddev=0.1, seed=SEED, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[Depth], dtype=tf.float32))

    # 全链接层网络设置
    fc1_weights = tf.Variable(tf.truncated_normal([29*4,64], stddev=0.1, seed=SEED, dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
    fc2_weights = tf.Variable(tf.truncated_normal([64, NUM_LABELS],stddev=0.1, seed=SEED, dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))

    def model(data, train=False):

        # 两层卷积池化
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool')

        # 全链接层
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]],name='feature')
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        # 随机drop 0.5 的神经元素，防止过拟合
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        logist = tf.matmul (hidden, fc2_weights) + fc2_biases
        return logist

    logits = model(train_data_node, True)

    # loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits))
    # 加L2 正则化项
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 1e-3 * regularizers

    batch = tf.Variable(0, dtype=tf.int32)
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(0.02, batch * BATCH_SIZE, train_size, 0.95, staircase=True)
    # Adam 下降法，最小化loss
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(loss, global_step=batch)
    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(model(eval_data),name='eval_prediction')


    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError ("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray (shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in xrange (0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run (eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run (eval_prediction, feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions


    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('Initialized!')

        # saver 保存模型
        saver = tf.train.Saver ()

        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {train_data_node: batch_data,
                       train_labels_node: batch_labels}
            sess.run(optimizer, feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                print('feature:')
                #print(feature)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print ('Validation error: %.1f%%' % error_rate(eval_in_batches (validation_data, sess), validation_labels))
                sys.stdout.flush()
                # Finally print the result!

                test_data1 = np.load (path + test_dataset[1] + '_train.npy')
                test_data1 =[calculate_ES.cal_es(x,window_size) for x in test_data1]
                test_labels1 = np.load (path + test_dataset[1] + '_label.npy')
                test_data2 = np.load (path+ test_dataset[2] + '_train.npy')
                test_data2=[calculate_ES.cal_es(x,window_size) for x in test_data2]
                test_labels2 = np.load (path + test_dataset[2] + '_label.npy')
                test_data1 = np.reshape (test_data1, [len(test_data1), 128, 1, 1])
                test_data2 = np.reshape (test_data2, [len(test_data2), 128, 1, 1])
                test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
                test_error1 = error_rate(eval_in_batches(test_data1, sess), test_labels1)
                test_error2 = error_rate(eval_in_batches(test_data2, sess), test_labels2)

                print('Test error: %.1f%%' % test_error)
                print('Test error: %.1f%%' % test_error1)
                print('Test error: %.1f%%' % test_error2)
                accuracy=(1-test_error/100)*100
                accuracy1=(1-test_error1/100)*100
                accuracy2=(1-test_error2/100)*100
                ava=(accuracy+accuracy1+accuracy2)/3.0
                print('test accuracy:%.1f%%' % accuracy)
                print('test accuracy:%.1f%%' % accuracy1)
                print('test accuracy:%.1f%%' % accuracy2)
                print('ava:%.1f%%' % ava)
        saver.save (sess, './scale-checkpoint_dir/MyModel', global_step=3500)


if __name__ == '__main__':
  tf.app.run(main=main)
