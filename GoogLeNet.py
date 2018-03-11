# !/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from models.Inception import inception_layer

class GoogLeNet(object):
    """Implementation of the GoogLeNet."""
    def __init__(self, input_data, num_label, keep_prob, 
                skip, model_path = 'DEFAULT', train_list = None
                ):
        self._num_label = num_label
        self._keep_prob = keep_prob
        self._skip = skip
        self._train_list = train_list
        if model_path == 'DEFAULT':
            self.model_path = './CNN/data/googlenet.npy'
        else:
            self.weigths_path = model_path

        self._create(input_data)
    
    def _create(self, input_data):
        '''Create GooogLeNet V1'''
        arg_scope = tf.contrib.framework.arg_scope
        # Conv1 7x7+2(S) -> MaxPool1 3x3+2(S) -> LRN1 
        # -> Conv2 1x1+1(V) -> Conv2 3x3+1(S) ->LRN2 -> Max_Pool2 3x3+2(S)
        with arg_scope([slim.conv2d, slim.max_pool2d],
                        stride = 2, padding = 'SAME',
                        ):
            train_able = self._istrain_able('conv1_7x7_s2')
            conv1 = slim.conv2d(input_data, 64, [7, 7], scope = 'conv1_7x7_s2', trainable = train_able)
            pool1 = slim.max_pool2d(conv1, [3, 3], scope = 'max_pool1_3x3_s2')
            lrn1 = tf.nn.lrn(pool1, depth_radius=2, alpha=2e-05, beta=0.75, name = 'lrn1')

            train_able = self._istrain_able('conv2_3x3_reduce')
            conv2_reduce = slim.conv2d(lrn1, 64, [1, 1], padding = 'VALID', 
                                        stride = 1, scope = 'conv2_3x3_reduce', trainable = train_able)
            train_able = self._istrain_able('conv2_3x3')
            conv2 = slim.conv2d(conv2_reduce, 192, [3, 3], stride = 1, scope = 'conv2_3x3', trainable = train_able)
            lrn2 = tf.nn.lrn(conv2, depth_radius=2, alpha=2e-05, beta=0.75, name = 'lrn2')
            pool2 = slim.max_pool2d(lrn2, [3, 3], scope = 'max_pool2_3x3_s2')

        # Inception3a -> Inception3b -> MaxPool3 3x3+2(S) -> Inception4a -> Inception4b -> Inception4c 
        # -> Inception4e -> MaxPool4 3x3+2(S) -> Inception5a -> Inception5b
        with arg_scope([inception_layer, slim.max_pool2d],
                        padding = 'SAME'
                        ):
            train_able = self._istrain_able('inception_3a')
            inception3a = inception_layer(pool2, 64, 96, 128, 16, 32, 32, name = 'inception_3a', trainable=train_able)
            train_able = self._istrain_able('inception_3b')
            inception3b = inception_layer(inception3a, 128, 128, 192, 32, 96, 64, name = 'inception_3b', trainable=train_able)
            pool3 = slim.max_pool2d(inception3b, [3, 3], scope = 'max_pool3_3x3_s2', stride = 2)

            train_able = self._istrain_able('inception_4a')
            inception4a = inception_layer(pool3, 192, 96, 208, 16, 48, 64, name = 'inception_4a', trainable = train_able)
            train_able = self._istrain_able('inception_4b')
            inception4b = inception_layer(inception4a, 160, 112, 224, 24, 64, 64, name = 'inception_4b', trainable = train_able)
            train_able = self._istrain_able('inception_4c')
            inception4c = inception_layer(inception4b, 128, 128, 256, 24, 64, 64, name = 'inception_4c', trainable = train_able)
            train_able = self._istrain_able('inception_4d')            
            inception4d = inception_layer(inception4c, 112, 144, 288, 32, 64, 64, name = 'inception_4d', trainable = train_able) 
            train_able = self._istrain_able('inception_4e')
            inception4e = inception_layer(inception4d, 256, 160, 320, 32, 128, 128, name = 'inception_4e', trainable = train_able)
            pool4 = slim.max_pool2d(inception4e, [3, 3], scope = 'max_pool4_3x3_s2', stride = 2)

            train_able = self._istrain_able('inception_5a')
            inception5a = inception_layer(pool4, 256, 160, 320, 32, 128, 128, name = 'inception_5a', trainable = train_able)
            train_able = self._istrain_able('inception_5b')
            inception5b = inception_layer(inception5a, 384, 192, 384, 48, 128, 128, name = 'inception_5b', trainable = train_able)

        
        # Branch1: Inception4a -> ArgPool_branch1 5x5+3(V) -> Conv_branch1 1x1+1(S) -> FC1_branch1 -> FC2_branch1 -> softmax0
        # Branch2: Inception4e -> ArgPool_branch2 5x5+3(V) -> Conv_branch2 1x1+1(S) -> FC1_branch2 -> FC2_branch2 -> softmax1
        # Branch3: Inception5b -> ArgPool_branch3 7x7+1(V) -> FC1_branch -> softmax2
        # 为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度。文章中说这两个辅助的分类器的loss应该加一个衰减系数，
        # 但看caffe中的model也没有加任何衰减。此外，实际测试的时候，这两个额外的softmax会被去掉。
        # only Branch3
        pool5 = slim.avg_pool2d(inception5b, [7, 7], stride = 1, padding = 'VALID', scope = 'arg_pool3_7x7_v1')
        dropout_layer = tf.nn.dropout(pool5, self._keep_prob, name = 'dropout')
        train_able = self._istrain_able('loss3_classifier')
        fc1 = slim.fully_connected(dropout_layer, self._num_label, scope = 'loss3_classifier')
        self.softmax = tf.nn.softmax(fc1, name = 'class_prob')

        
    
    def loadModel(self, sess):
        wDict = np.load(self.model_path, encoding = 'bytes').item()
        for name in wDict:
           if name not in self._skip:
                for p in wDict[name]:
                    if len(wDict[name][p].shape) == 1:
                        # sess.run(tf.get_variable('biases:0'), trainable = False). assign((p))
                        assign_op, feed_dict_init = slim.assign_from_values({name +'/biases:0' : wDict[name][p]})
                    else:
                        # sess.run(tf.get_variable('weigths:0'), trainable = False).assgin((p))
                        assign_op, feed_dict_init = slim.assign_from_values({name +'/weights:0' : wDict[name][p]})
                        
                    sess.run(assign_op, feed_dict_init)
    
    def _istrain_able(self, layer_name):
        if self._train_list == None:
            return True

        if layer_name in self._train_list:
            return True
        else:
            return False

    def get_prediction(self):
        return self.softmax