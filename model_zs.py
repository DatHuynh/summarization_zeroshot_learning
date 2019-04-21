# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:51:05 2019

@author: badat
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from utils import get_init_embedding
from nltk.corpus import stopwords
import numpy as np
import pdb

class Model(object):
    
    def construct_w2v_features(self):
        with tf.variable_scope("zeroshot_text_feature"):
            print("zeroshot_text_feature")
            self.softmaxs = tf.nn.softmax(self.logits_reshape,axis=2)
            self.frequency = tf.reduce_sum(self.softmaxs,axis = 1)
            ### mask out stopwords ###
            idx_non_stopwords = [i for i in range(self.vocabulary_size) if self.reversed_dict[i] not in stopwords.words('english')]
#            mask = tf.reshape(tf.one_hot(idx_non_stopwords,self.vocabulary_size,on_value=1.0,off_value=0.0,dtype=tf.float32),[-1,self.vocabulary_size])
#            mask = tf.reduce_sum(mask,axis=0)[tf.newaxis,:]
            mask = np.zeros((1,self.vocabulary_size))
            mask[0,idx_non_stopwords]=1
            self.frequency = tf.multiply(self.frequency,mask)
            ### mask out stopwords ###
            _,self.subset_word_ids = tf.nn.top_k(self.frequency,k=10)
            text_features = tf.matmul(self.frequency,self.embeddings)
            text_features = tf.nn.l2_normalize(text_features,1)
            return text_features
    
    def zeroshot_module(self):
        with tf.variable_scope("zeroshot_text_feature"):
            print("zeroshot_text_feature")
            W = tf.get_variable('W',shape=[2048,300])
            self.preds=tf.matmul(self.image_features,tf.matmul(W,tf.transpose(self.text_features)))
            _,self.idx_preds = tf.nn.top_k(self.preds,k=3)
            mask = tf.reshape(tf.one_hot(self.labels,self.n_class,on_value=True,off_value=False),[-1,self.n_class])
            preds_target = tf.boolean_mask(self.preds,mask)
            preds_max = tf.reduce_max(self.preds,axis = 1)
            loss = tf.reduce_mean(tf.maximum(1+preds_max-preds_target,0))
            return loss
    
    def combine_text_features(self):
        with tf.variable_scope("zeroshot_combine_text_features"):
            print("zeroshot_combine_text_features alpha_summary {}".format(self.alpha_summary))
            return self.alpha_summary*self.text_features+(1-self.alpha_summary)*self.w2v
    
    def optimization(self):
        with tf.name_scope("optimizer"):
            print("optimizer")
            if self.opt_var_type == 0:
                print('optimize joint embedding only')
                self.params = [p for p in tf.trainable_variables() if 'zeroshot' in p.name]
            elif self.opt_var_type == 1:
                print('optimize joint embedding + decoder')
                self.params = [p for p in tf.trainable_variables() if ('zeroshot' in p.name) or ('decoder' in p.name)]
            elif self.opt_var_type == 2:
                print('optimize all')
                self.params = tf.trainable_variables()
            
            print("RMSProp")
            gradients = tf.gradients(self.loss, self.params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#            self.lr_decay = tf.train.cosine_decay_restarts(self.learning_rate,self.global_step,200)
            self.lr_decay =  tf.train.exponential_decay(self.learning_rate,self.global_step,1000,0.5,staircase=True)
#            self.lr_decay = tf.constant(self.learning_rate)
            optimizer = tf.train.RMSPropOptimizer(self.lr_decay)
            self.update = optimizer.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
            
#            optimizer = tf.train.RMSPropOptimizer(self.lr_decay)
#            grad_vars = optimizer.compute_gradients(self.loss,self.params)
#            self.update = optimizer.apply_gradients(grad_vars,global_step=self.global_step)
    
    def __init__(self, reversed_dict, article_max_len, summary_max_len, args, w2v, alpha_summary, opt_var_type):
        print('-'*30)
        self.vocabulary_size = len(reversed_dict)
        self.reversed_dict = reversed_dict
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = args.beam_width
        self.n_class = 50
        self.w2v = w2v
        self.keep_prob = 1.0
        self.cell = tf.nn.rnn_cell.BasicLSTMCell
        self.alpha_summary = alpha_summary
        self.opt_var_type = opt_var_type
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, article_max_len])
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])
        self.global_step = tf.Variable(0, trainable=False)
        self.image_features = tf.placeholder(name='image_features',dtype=tf.float32, shape=[None, 2048])
        self.labels = tf.placeholder(name='labels',dtype=tf.int32, shape=[None])
        with tf.name_scope("embedding"):
#            if not forward_only and args.glove:
#                init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.embedding_size), dtype=tf.float32)
#            else:
            init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X), perm=[1, 0, 2])
            self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input), perm=[1, 0, 2])

        with tf.name_scope("encoder"):
            fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell) for cell in bw_cells]

            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.encoder_emb_inp,
                sequence_length=self.X_len, time_major=True, dtype=tf.float32)
            self.encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = self.cell(self.num_hidden * 2)

            attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len, normalize=True)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.num_hidden * 2)
            initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            initial_state = initial_state.clone(cell_state=self.encoder_state)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3))
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope,maximum_iterations = summary_max_len)
            self.decoder_output = outputs.rnn_output
            self.logits = tf.transpose(
                self.projection_layer(self.decoder_output), perm=[1, 0, 2])     #tensors with second dimension [dim 1] is the number of words in the summarization and last dim is logits score of each predicted words
            self.logits_reshape = tf.concat(
                [self.logits, tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size])], axis=1)
        
        self.summary_preds= tf.argmax(self.logits,axis=2)
        
        self.text_features = self.construct_w2v_features()
        self.text_features = self.combine_text_features()
        self.loss = self.zeroshot_module()
        
        self.optimization()
        print('-'*30)