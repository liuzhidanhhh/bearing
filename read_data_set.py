# cnn 网络中用于读取数据的函数

#encoding:utf-8
import numpy as np

class DataSet(object):
  def __init__(self, images, labels):
    images=np.array(images)
    labels=np.array(labels)
    assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
def read_data_set(dataname,labelname):
  class DataSets(object):
    pass
  data_sets=DataSets()
  data=np.load(dataname)
  label=np.load(labelname)
  def one_hot(label):
    label_one_hot = np.zeros((len(label), 4))
    offset = np.arange(len(label)) * 4
    label_one_hot.flat[offset + label] = 1
    return label_one_hot

  #label = one_hot(label)
  data=np.array(data)
  label=np.array(label)
  train0 = []
  train1 = []
  train2 = []
  train3 = []
  label0 = []
  label1 = []
  label2 = []
  label3 = []

  for i in range(len(data)):
    if i%4==0:
      train0.append(data[i])
      label0.append(label[i])
    elif i%4==1:
      train1.append (data[i])
      label1.append (label[i])
    elif i%4==2:
      train2.append (data[i])
      label2.append (label[i])
    else:
      train3.append (data[i])
      label3.append (label[i])

  data_sets.train0=DataSet(train0,label0)
  data_sets.train1=DataSet(train1,label1)
  data_sets.train2=DataSet(train2,label2)
  data_sets.train3=DataSet(train3,label3)

  return data_sets
#data=read_data_set('./1024-data/1730_train.npy','./1024-data/1730_label.npy')