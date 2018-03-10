# !/usr/bin/python3
# -*- coding: utf-8 -*-
# File: Inception.py
# Author: starsingchow <starstarstarchow@gmail.com>
# Implement: https://github.com/conan7882/GoogLeNet-Inception-tensorflow/blob/master/lib/models/inception.py

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import add_arg_scope

@add_arg_scope
def inception_layer(inputs,
                    conv_11_size,
                    conv_33_reduce_size, conv_33_size,
                    conv_55_reduce_size, conv_55_size,
                    pool_size,
                    name = 'inception',
                    stride = 1,
                    padding = 'SAME',
                    trainable = True
                    ):
    arg_scope = tf.contrib.framework.arg_scope

    with arg_scope([slim.conv2d],stride = stride, padding = padding, trainable = trainable):
        with arg_scope([slim.max_pool2d], stride = stride, padding = padding):
            # Conv 1x1+1(S)
            conv_11 = slim.conv2d(inputs, conv_11_size, [1, 1], scope = '{0}_1x1'.format(name))
        
            # Conv 1x1+1(S) -> Conv 3x3+1(S)
            conv_33_reduce = slim.conv2d(inputs, conv_33_reduce_size, [1, 1], scope = '{0}_3x3_reduce'.format(name))
            conv_33 = slim.conv2d(conv_33_reduce, conv_33_size, [3, 3], scope = '{0}_3x3'.format(name))

            # Conv 1x1+1(S) -> Conv 5x5+1(S)
            conv_55_reduce = slim.conv2d(inputs, conv_55_reduce_size, [1, 1], scope = '{0}_5x5_reduce'.format(name))
            conv_55 = slim.conv2d(conv_55_reduce, conv_55_size, [5, 5], scope ='{0}_5x5'.format(name))

            # MaxPool 3x3+1(S) -> Conv 1x1+1(S)
            pool = slim.max_pool2d(inputs, [3, 3], scope = '{0}_pool'.format(name))
            convpool = slim.conv2d(pool, pool_size, [1, 1], scope = '{0}_pool_proj'.format(name))

    return tf.concat([conv_11, conv_33, conv_55, convpool], 3, name = '{0}_concat'.format(name))
