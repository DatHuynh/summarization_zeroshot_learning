# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:10:18 2018

@author: badat
"""
train_path = './TFRecord/zs_train_AWA2_feature.tfrecords'
test_path = './TFRecord/zs_test_AWA2_feature.tfrecords'
#%%
batch_size = 32#32
learning_rate_base = 0.001
report_interval=10
n_cycles = 1000
e2e_n_cycles = 1000