# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 03:00:49 2019

@author: badat
"""

import tensorflow as tf
import pickle
from model_zs import Model
from utils import build_dict, build_dataset, batch_iter,build_wiki_text,decode_id
import global_setting_AWA2
import os
import numpy as np
import pandas as pd
import pdb
#%% Define experimential setting
idx_GPU=0
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
is_save = False
n_iterations = 2000
article_max_len = 200
summary_max_len = 50
alphas = [0,0.2,0.4,0.6,0.8,1]
batch_size = 32
w2v_model = 'glove_vecs'
path_w2v = './data/{}.npy'.format(w2v_model)
#docker_path = '/home/project_amadeus/'
#save_dir = os.path.join(docker_path,'/mnt/raptor/hbdat/summarized_zs/'+w2v_model+'_alpha_{}')
save_dir = './results/'+w2v_model+'/alpha_{}'
#%% load wiki data
w2v=np.load(path_w2v)
n_class = w2v.shape[0]
n_w_dim = w2v.shape[1]
w2v = w2v.astype(np.float32)

with open("args.pickle", "rb") as f:
    args = pickle.load(f)
    args.learning_rate=0.001

with open("./wiki_data/wiki_article.pkl","rb") as f:
    wiki_data = pickle.load(f)
    classes = [d[0] for d in wiki_data]
    wiki_data = [d[1] for d in wiki_data]
    
print("Loading dictionary...")
word_dict, reversed_dict, _, _= build_dict("valid", args.toy)

print("Loading validation dataset...")
wiki_context = build_wiki_text(wiki_data, word_dict, article_max_len)
string_context = decode_id(wiki_context,reversed_dict)
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
dataset_temp = tf.data.TFRecordDataset(global_setting_AWA2.train_path)
dataset_temp = dataset_temp.map(parser).batch(50000)
(_,_,labels_temp,_) = dataset_temp.make_one_shot_iterator().get_next()
(labels_temp)=sess.run([labels_temp])
seen_classes = np.unique(labels_temp)
unseen_classes = np.array([i for i in range(50) if i not in seen_classes])

dataset = tf.data.TFRecordDataset(global_setting_AWA2.trainval_path)
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
#%% evaluation function
def evaluate_macro(idx_preds_test_v,labels_tst):
    count = 0.0
    for idx,l in enumerate(labels_tst):
        if l in idx_preds_test_v[idx]:
            count+=1.0
    return count /len(labels_tst)

def evaluate_micro(idx_preds_test_v,labels_tst):
    classes = np.unique(labels_tst)
    acc_per_class=np.zeros(len(classes))
    indicator = np.zeros(idx_preds_test_v.shape[0])
    
    for idx,l in enumerate(labels_tst):
        if l in idx_preds_test_v[idx]:
            indicator[idx]=1.0
            
    for idx_c,c in enumerate(classes):
        mask = labels_tst==c
        acc_per_class[idx_c]=np.mean(indicator[mask])
    
    acc_all = np.mean(acc_per_class)
    acc_seen = np.mean(acc_per_class[seen_classes])
    acc_unseen = np.mean(acc_per_class[unseen_classes])
    return acc_all,acc_seen,acc_unseen,acc_per_class
#%% model definition
alpha = tf.Variable(0.0,name='alpha',dtype = tf.float32, trainable=False)
with tf.variable_scope(tf.get_variable_scope()):
    model = Model(reversed_dict, article_max_len, summary_max_len, args, w2v=w2v,alpha_summary=alpha,opt_var_type=0)
wiki_context = np.array(wiki_context)
context_len = [len([y for y in x if y != 0]) for x in wiki_context]
#%% [Qualitative result] get important keywords for each classes
restored_vars = [p for p in tf.trainable_variables() if 'zeroshot' not in p.name]
saver = tf.train.Saver(restored_vars)
ckpt = tf.train.get_checkpoint_state("./saved_model/")

def get_keywords():
    wiki_keywords = []
    feed_dict = {
                model.batch_size: len(wiki_context),
                model.X: wiki_context,
                model.X_len: context_len,
                model.image_features:fs_tst
            }
    prediction = sess.run(model.subset_word_ids, feed_dict=feed_dict)
    prediction_output = [[reversed_dict[y] for y in x] for x in prediction]
    for idx,line in enumerate(prediction_output):
        keywords = " ".join(line)
        wiki_keywords.append(keywords)
    return wiki_keywords
#%% Run experiment on different alphas
for alpha_v in alphas:
    print('-'*30)
    print('alpha {}'.format(alpha_v))
    save_path=save_dir.format(alpha_v)
    os.makedirs(save_path)
    
    df = pd.DataFrame()
    df['classes'] = classes
    
    tf.global_variables_initializer().run()
    sess.run(iterator.initializer)
    saver.restore(sess, ckpt.model_checkpoint_path)
    #%%
    
    df['keywords_before'] = get_keywords()
    
    sess.run(alpha.assign(alpha_v))
    
    for i in range(n_iterations):
        ids_tr_v,fs_tr_v,labels_tr_v=sess.run([ids_tr,fs_tr,labels_tr])
        
        feed_dict = {
            model.batch_size: len(wiki_context),
            model.X: wiki_context,
            model.X_len: context_len,
            model.image_features:fs_tr_v,
            model.labels:labels_tr_v
        }
        
    #    pdb.set_trace()
        
        _, step, loss, lr_decay_v = sess.run([model.update, model.global_step, model.loss, model.lr_decay], feed_dict=feed_dict)
        
        if i %100 == 0 or i == n_iterations-1:
            feed_dict = {
                model.batch_size: len(wiki_context),
                model.X: wiki_context,
                model.X_len: context_len,
                model.image_features:fs_tst
            }
            idx_preds_tst=sess.run(model.idx_preds,feed_dict=feed_dict)
            acc_macro = evaluate_macro(idx_preds_tst,labels_tst)
            acc_all,acc_seen,acc_unseen,acc_per_class = evaluate_micro(idx_preds_tst,labels_tst)
    #        pdb.set_trace()
            print(i,acc_macro,acc_unseen,lr_decay_v)
            print(alpha.eval())
            
    df['keywords_after'] = get_keywords()
    df['acc']=acc_per_class
    if is_save:
        df.to_csv(os.path.join(save_path,'result.csv'))
        saver.save(sess, os.path.join(save_path,"saved_model/model.ckpt"))
    #    print('Summaries are saved to "result.txt"...')
