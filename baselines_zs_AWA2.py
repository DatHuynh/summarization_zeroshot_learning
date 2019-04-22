# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:21:24 2019

@author: badat
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import os.path
import os
import numpy as np
import time
import pdb
#from measurement import apk,compute_number_misclassified
from D_utility import project_signature,zeroshot_evaluation,LoadLabelMap,preprocessing_graph,signature_completion,evaluate_completion
import global_setting_AWA2
#%% logging level
tf.logging.set_verbosity(tf.logging.INFO)
learning_rate_base=0.001
batch_size=32
#%% data flag
idx_GPU=1
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
path_w2v = './data/glove_vecs.npy'
#%% load embedding data
w2v=np.load(path_w2v)
n_class = w2v.shape[0]
n_w_dim = w2v.shape[1]
w2v = np.transpose(w2v).astype(np.float32)
#%% define TFRecord parse for high capacity data loading
def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string),
               'attribute':tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id = tf.decode_raw( parsed['img_id'],tf.int64)
    feature = tf.decode_raw( parsed['feature'],tf.float32)
    label = tf.squeeze(tf.decode_raw(parsed['label'],tf.int32))
    attribute = tf.decode_raw( parsed['attribute'],tf.int32)    
    return img_id,feature,label,attribute
#%%
sess = tf.InteractiveSession()
#%% load training data
dataset = tf.data.TFRecordDataset(global_setting_AWA2.train_path)
dataset = dataset.map(parser)
dataset = dataset.shuffle(20000)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
(ids_tr,fs_tr,labels_tr,attrs_tr) = iterator.get_next()
#%% load testing data
dataset_tst = tf.data.TFRecordDataset(global_setting_AWA2.test_path)
dataset_tst = dataset_tst.map(parser).batch(50000)
(ids_tst,fs_tst,labels_tst,attrs_tst) = dataset_tst.make_one_shot_iterator().get_next()
(ids_tst,fs_tst,labels_tst,attrs_tst)=sess.run([ids_tst,fs_tst,labels_tst,attrs_tst])
#%% 
W = tf.get_variable('W',shape=[2048,n_w_dim])   # embedding function
preds=tf.matmul(fs_tr,tf.matmul(W,w2v))         
mask = tf.reshape(tf.one_hot(labels_tr,n_class,on_value=True,off_value=False),[-1,n_class])
preds_target = tf.boolean_mask(preds,mask)
preds_max = tf.reduce_max(preds,axis = 1)       # maximum prediction score
loss = tf.reduce_mean(tf.maximum(1+preds_max-preds_target,0)) # loss
#%% optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate_base)
grad_vars = optimizer.compute_gradients(loss)
train = optimizer.apply_gradients(grad_vars)
#%%
_,idx_preds_test = tf.nn.top_k(tf.matmul(fs_tst,tf.matmul(W,w2v)),k=3)  # top 3 accuracy error
#%% evaluation
def evaluate(idx_preds_test_v,labels_tst):
    count = 0.0
    for idx,l in enumerate(labels_tst):
        if l in idx_preds_test_v[idx]:
            count+=1.0
    return count /len(labels_tst)
#%% training
tf.global_variables_initializer().run()
sess.run(iterator.initializer)
for i in range(1000):
    if i%100==0:
        print(i)
    _,l,idx_preds_test_v=sess.run([train,loss,idx_preds_test])
    acc = evaluate(idx_preds_test_v,labels_tst)#np.sum(idx_preds_test_v-labels_tst == 0)/len(idx_preds_test_v)
print('top 3 accuracy: ',acc)
#%%
sess.close()
tf.reset_default_graph()