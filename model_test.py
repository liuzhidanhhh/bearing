import conv_model
import numpy as np

# test data 30:30:30:30
# data1

train_data='model_data/data7.npy'
train_label='model_data/label7.npy'
test_data='model_data/test.npy'
test_label='model_data/test_label.npy'
conv_model.main(train_data,train_label,test_data,test_label,3000)
