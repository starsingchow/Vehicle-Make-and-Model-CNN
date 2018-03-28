"""

"""
import tensorflow as tf
import numpy as np

class AlexNet(object):
    """Implementation of the AlexNet."""
    def __init__(self, input_data, num_label, keep_prob, skip, 
                weights_path='DEFAULT', train_list = None, regularizer = None):
        '''inital data'''
        self._input_data = input_data
        self._num_label = num_label
        self._keep_prob = keep_prob
        self._skip = skip
        self._train_list = train_list
        self._regularizer = regularizer
        if weights_path == 'DEFAULT':
            self.weights_path = './Vehicle-Make-and-Model-CNN/data/bvlc_alexnet.npy'
            # self.weights_path = './CNN/data/bvlc_alexnet.npy'
        else:
            self.weights_path = weights_path
        self.create()

    def create(self):
        '''Create AlexNet'''
        # conv_layer_1: Conv(w relu) -> lrn -> max_pool
        train_able = self._istrain_able('conv1')
        conv1 = self._conv(self._input_data, 11, 11, 96, 4, 4, 'conv1', padding = 'VALID', trainable=train_able)
        norm1 = self._lrn(conv1, 2, 1e-04, 0.75, name = 'norm1')
        pool1 = self._max_pool(norm1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')

        # conv_layer_2: Conv(w relu groups = 2) -> lrn -> max_pool
        train_able = self._istrain_able('conv2')
        conv2 = self._conv(pool1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2', trainable = train_able)
        norm2 = self._lrn(conv2, 2, 1e-04, 0.75, name = 'norm2')
        pool2 = self._max_pool(norm2, 3, 3, 2, 2, padding = 'VALID', name = 'pool2')

        # conv_layer_3: Conv(w relu)
        train_able = self._istrain_able('conv3')
        conv3 = self._conv(pool2, 3, 3, 384, 1, 1, name = 'conv3', trainable = train_able)
        
        #conv_layer_4: Conv(w relu groups = 2)
        train_able = self._istrain_able('conv4')
        conv4 = self._conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4', trainable = train_able)

        #conv_layer_5: conv(w relu groups = 2) -> max_pool
        train_able = self._istrain_able('conv5')
        conv5 = self._conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5', trainable = train_able)
        pool5 = self._max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

        #FC_layer_6: Flatten -> FC(w relu) -> dropout
        # size = int(pool5.shape[1] * pool5.shape[2] * pool5.shape[3])
        flattened = tf.reshape(pool5, [-1, 256 * 6 * 6])
        train_able = self._istrain_able('fc6')
        fc6 = self._fc(flattened, 256 * 6 * 6, 4096, name = 'fc6', trainable = train_able)
        dropout6 = self._dropout(fc6, self._keep_prob)

        #FC_layer_7: FC(w relu) -> dropout
        train_able = self._istrain_able('fc7')
        fc7 = self._fc(dropout6, 4096, 4096, name = 'fc7', trainable = train_able)
        dropout7 = self._dropout(fc7, self._keep_prob)

        #FC_layer_8: FC(w relu) -> dropout
        train_able = self._istrain_able('fc8')
        self.fc8 = self._fc(dropout7, 4096, self._num_label, name = 'fc8', relu=False, trainable = train_able)

    def loadModel(self, sess):
        print('--load ImageNet Model--')
        print('load dir: {0}'.format(self.weights_path))
        wDict = np.load(self.weights_path, encoding = 'bytes').item()
        for name in wDict:
            if name not in self._skip:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            sess.run(tf.get_variable('biases', trainable=False).assign(p))
                        else:
                            sess.run(tf.get_variable('weights', trainable=False).assign(p))


    def _conv(self, x, filter_height, filter_weight, num_filters, stride_y, stride_x, name, padding = 'SAME', groups = 1, trainable = True):
        input_channel = int(x.get_shape()[-1])
        
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1], padding = padding)
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable("weights", shape=[filter_height,
                                                        filter_weight,
                                                        input_channel/groups,
                                                        num_filters],
                                                        trainable = trainable
                                                        )   
            biases = tf.get_variable("biases", shape = [num_filters], trainable = trainable)

            input_groups = tf.split(x, num_or_size_splits = groups, axis = 3)
            weights_groups = tf.split(weights, num_or_size_splits = groups, axis = 3)

            conv_groups = [convolve(i, j) for i, j in zip(input_groups, weights_groups)]
            conv = tf.concat(conv_groups, axis = 3)

            bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv), name = 'bias')
            relu = tf.nn.relu(bias, name = 'relu')
        return relu

    def get_prediction(self):
        softmax = tf.nn.softmax(self.fc8)
        return softmax

    def _max_pool(self, x, filter_weight, filter_height, stride_y, stride_x, name, padding = 'SAME'):
        return tf.nn.max_pool(x, ksize = [1, filter_height, filter_weight, 1], strides = [1, stride_y, stride_x, 1], 
                              padding = padding, name=name)

    def _lrn(self, x, depth_radius, alpha, beta, name, bias = 1.0):
        return tf.nn.lrn(x, depth_radius = depth_radius, bias = bias, alpha = alpha, beta = beta, name = name)

    def _fc(self, x, num_in, num_out, name, relu = True, trainable = True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable("weights", shape=[num_in, num_out], 
                                        trainable=trainable, 
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=2.0)
                                        )
            biases = tf.get_variable("biases", shape = [num_out], 
                                        trainable=trainable,
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=2.0)
                                        )

        if self._regularizer != None:
            tf.add_to_collection('losses', self._regularizer(weights))

        if relu:
            return tf.nn.relu(tf.matmul(x, weights) + biases)
        else:
            return tf.matmul(x, weights) + biases

    def _dropout(self, x, keep_prob, name = None):
        return tf.nn.dropout(x, keep_prob, name)

    def _istrain_able(self, layer_name):
        if self._train_list == None:
            return True

        if layer_name in self._train_list:
            return True
        else:
            return False