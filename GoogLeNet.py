import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from models.Inception import inception_layer


class GoogLeNet(object):
    """Implementation of the GoogLeNet."""
    def __init__(self, trainable, weights_path = 'DEFAULT'):
        
        self._trainable = trainable
        if weights_path == 'DEFAULT':
            self.weights_path = './CNN/model/'
        else:
            self.weigths_path = weights_path
        self._create()
    
    def _create(self, inputs, data_dict):
        '''Create GooogLeNet V1'''
        arg_scope = tf.contrib.framework.arg_scope
        # Conv1 7x7+2(S) -> MaxPool1 3x3+2(S) -> LRN1 
        # -> Conv2 1x1+1(V) -> Conv2 3x3+1(S) ->LRN2 -> Max_Pool2 3x3+2(S)
        with arg_scope([slim.conv2d, slim.max_pool2d],
                        stride = 2, padding = 'SAME',
                        activation_fn = tf.nn.relu, 
                        data_dict = data_dict
                        ):
            conv1 = slim.conv2d(inputs, 64, [7, 7], scope = 'conv1_7x7_s2')
            pool1 = slim.max_pool2d(conv1, [3, 3], scope = 'max_pool1_3x3_s2')
            lrn1 = tf.nn.lrn(pool1, depth_radius=2, alpha=2e-05, beta=0.75, name = 'lrn1')

            conv2_reduce = slim.conv2d(lrn1, 64, [1, 1], padding = 'VALID', 
                                        stride = 1, scope = 'conv2_3x3_reduce')
            conv2 = slim.conv2d(conv2_reduce, 192, [3, 3], scope = 'conv2_3x3')
            lrn2 = tf.nn.lrn(conv2, depth_radius=2, alpha=2e-05, beta=0.75, name = 'lrn2')
            pool2 = slim.max_pool2d(lrn2, [3, 3], scope = 'max_pool2_3x3_s2')

        # Inception_3a -> Inception_3b 
        with arg_scope([inception_layer],
                        trainable =  
                        data_dict = data_dict
                        )
            inception3a = inception_layer(pool2, 64, 96, 128, 16, 32, 32)
            