from collections import namedtuple

import tensorflow as tf
import numpy as np


'''
tf.contrib.layers.separable_conv2d
normalizer_fn: Normalization function to use instead of biases. If normalizer_fn is provided then biases_initializer and biases_regularizer are ignored and biases are not created nor added. default set to None for no normalizer function
'''
slim  = tf.contrib.slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSeqConv', ['kernel', 'stride', 'depth'])

_CONV_DEFS = [
    Conv(kernel=[3, 3], stride = 2, depth = 32),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 64),
    DepthSepConv(kernel = [3, 3], stride = 2, depth = 128),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 128),
    DepthSepConv(kernel = [3, 3], stride = 2, depth = 256),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 256),
    DepthSepConv(kernel = [3, 3], stride = 2, depth = 512),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 512),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 512),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 512),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 512),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 512),
    DepthSepConv(kernel = [3, 3], stride = 2, depth = 1024),
    DepthSepConv(kernel = [3, 3], stride = 1, depth = 1024)
]

batch_norm_params = {
      'is_training': True,
      'center': True,
      'scale': True,
      'decay': 0.9997,
      'epsilon': 0.001,
  }

class MobileNets(object):
    def __init__(self, inputs, num_label, keep_prob, skip, model_path = 'DEFAULT', conv_defs = None):
        self._num_label = num_label
        self._keep_prob = keep_prob
        self._skip = skip
        if model_path == 'DEFAULT':
            self._model_path = './CNN/data/mobilenets_v1_1.0_224.npy'
        else:
            self._model_path = model_path

        if conv_defs == None:
            conv_defs = _CONV_DEFS

        self.prediction = self.create(inputs, conv_defs)
        
    
    def create(self, inputs, conv_defs):
        with tf.variable_scope('MobilenetV1', 'MobilenetV1', [inputs]) as scope:
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                net, end_points = self.mobilenets_base(inputs, conv_defs)
                
                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, [7, 7], stride = 1, padding = 'VALID', scope = 'AvgPool_1a')
                    print(net.shape)
                    end_points['AvgPool_1a'] = net
                
                    net = slim.dropout(net, keep_prob = self._keep_prob, scope = 'Dropout_1b', is_training = True)
                
                    logits = slim.conv2d(net, self._num_label, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'Conv2d_1c_1x1')
                    # prediction = slim.softmax(logits, scope = 'Predictions')
                print(logits.shape)
                prediction = tf.nn.softmax(logits, name = 'class_prob')
                end_points['Predictions'] = prediction
                
        return prediction
                
        
    def mobilenets_base(self, inputs, conv_defs, use_explicit_padding = False, scope = None, final_endpoint = 'Conv2d_13_pointwise'):
        padding = 'SAME'
        end_points = {}

        if use_explicit_padding:
            padding = 'VALID'
        with tf.variable_scope(scope, 'Mobilenets', [inputs]):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding = padding):
                net = inputs
                for i, conv_def in enumerate(conv_defs):
                    end_point_base = 'Conv2d_%d' % i
                    
                    if isinstance(conv_def, Conv):
                        end_point = end_point_base

                        net = slim.conv2d(net, conv_def.depth, conv_def.kernel, 
                                            stride = conv_def.stride, 
                                            normalizer_fn = slim.batch_norm,
                                            scope = end_point)
                        
                        end_points[end_point] = net

                        if end_point == final_endpoint:
                            return net, end_points

                    elif isinstance(conv_def, DepthSepConv):
                        end_point = end_point_base + '_depthwise'

                        net = slim.separable_conv2d(net, None, conv_def.kernel,
                                            stride = conv_def.stride,
                                            depth_multiplier = 1,
                                            normalizer_fn = slim.batch_norm,
                                            scope = end_point)

                        end_points[end_point] = net
                        if end_point == final_endpoint:
                            return net, end_points

                        end_point = end_point_base + '_pointwise' 

                        net = slim.conv2d(net, conv_def.depth, [1, 1], 
                                            stride = 1, 
                                            normalizer_fn = slim.batch_norm,
                                            scope = end_point)
                        end_points[end_point] = net
                        if end_point == final_endpoint:
                            return net, end_points
    
    def loadModel(self, sess):
        wDict = np.load(self._model_path, encoding = 'bytes').item()
        for name in wDict:
            if name not in self._skip:
                for p in wDict[name]:
                    if p == 'BatchNorm':
                        for k in wDict[name][p]:
                            assign_op, feed_dict_init = slim.assign_from_values({'MobilenetV1/Mobilenets/' + name + '/' + p + '/' + k + ':0': wDict[name][p][k]})
                    elif  p == 'Conv2d_1c_1x1':
                        for k in wDict[name][p]:
                            assign_op, feed_dict_init = slim.assign_from_values({'MobilenetV1/' + name + '/' + p + '/' + k + ':0': wDict[name][p][k]})
                    else:
                        assign_op, feed_dict_init = slim.assign_from_values({'MobilenetV1/Mobilenets/' + name +'/' + p +':0' : wDict[name][p]})
                       
                    sess.run(assign_op, feed_dict_init)

    def get_prediction(self):
        return self.prediction