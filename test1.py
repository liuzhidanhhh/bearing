# cnn 架构代码
# 测试用的网络
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import sys
import time
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import calculate_ES


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
path='512-scale-data'
train_dataset='1730'
test_dataset=['1797','1750','1772']
point=0
num=32
def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])


def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph ('./scale-checkpoint_dir/MyModel-3500.meta')
        saver.restore (sess, tf.train.latest_checkpoint ('./scale-checkpoint_dir'))
        graph = tf.get_default_graph()
        eval_data=graph.get_tensor_by_name('eval_data:0')
        eval_prediction = graph.get_tensor_by_name ("eval_prediction:0")
        #feature=graph.get_tensor_by_name('feature:0')


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


        #测试不同转速的数据集
        test_data = np.load('512-scale-data/1797_train.npy')
        test_labels1 = np.load('512-scale-data/1797_label.npy')
        test_data1=[calculate_ES.cal_es(x, window_size) for x in test_data]
        test_data1 = np.reshape(test_data1, [len(test_data1), 128, 1, 1])
        test_error1 = error_rate(eval_in_batches(test_data1, sess), test_labels1)
        print('Test error: %.1f%%' % test_error1)
        accuracy1 = (1 - test_error1 / 100) * 100
        print('test accuracy:%.1f%%' % accuracy1)

        # 测试GAN 生成的数据集
        # test_data = np.load ('scale-gen-512-data/1730_gen_data2.npy')
        # test_labels = np.load ('scale-gen-512-data/1730_gen_label2.npy')
        # for test,label in zip(test_data,test_labels):
        #     data=[calculate_ES.cal_es(x, window_size) for x in test]
        #     data = np.reshape (data , [len (data), 128, 1, 1])
        #     error = error_rate (eval_in_batches (data, sess), label)
        #     print(numpy.argmax(eval_in_batches (data, sess), 1))
        #     print ('Test error: %.1f%%' % error)
        #     accuracy1 = (1 - error/ 100) * 100
        #     print ('test accuracy:%.1f%%' % accuracy1)

if __name__ == '__main__':
    main()

