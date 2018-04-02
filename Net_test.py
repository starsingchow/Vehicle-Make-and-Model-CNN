import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from AlexNet import AlexNet
from GoogLeNet import GoogLeNet
from MobileNets import MobileNets
from MobileNets_depth import MobileNets_depth

from input_data import get_test_data

import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('net_model', choices = ['alexnet', 'googlenet','mobilenet', 'mobilenet_0.75', 'mobilenet_0.5'], default='folder', help='choose net')
parser.add_argument('train_model', choices = ['finetune', 'fulltrain','parttune'], default='folder', help='choose net')
parser.add_argument('--data_dir', type=str, default='', help ='test Data Dir')
parser.add_argument('--model_dir', type=str, default='', help ='Model Dir')

FLAGS, _ = parser.parse_known_args()

args = parser.parse_args()
NET_TYPE = args.net_model
TRAIN_MODEL = args.train_model
DATA_PATH = args.data_dir
MODEL_DIR = args.model_dir
NUMBER_CHANNEL = 3
IMAGE_SIZE = 224
MOVING_AVERAGE_DECAY = 0.99

if NET_TYPE == 'alexnet':
    IMAGE_SIZE = 227
model_dir = os.path.join(MODEL_DIR, NET_TYPE, TRAIN_MODEL)

def evaluate(net,train_list):
    k = 0
    rate_top_1_save = []
    rate_top_5_save = []
    y_true = []
    y_predict = []
    while k<1:
        tf.reset_default_graph()
        with tf.Graph().as_default() as g:
            data_iterator = get_test_data(DATA_PATH, 473)
            next_element = data_iterator.get_next()
            x = tf.placeholder(
                tf.float32,
                [473, IMAGE_SIZE, 
                IMAGE_SIZE, NUMBER_CHANNEL],
                name = 'input-x'
                )
            y_ = tf.placeholder(
                tf.int64, 
                [None],
                name = 'input-y'
            )

            if NET_TYPE == 'mobilenet_0.5':
                model = net(x, 196, 1, None, train_list=train_list, depth_multiplier=0.5)
            elif NET_TYPE == 'mobilenet_0.75':
                model = net(x, 196, 1, None, train_list=train_list, depth_multiplier=0.75)
            else:
                model = net(x, 196, 1, None, train_list=train_list)
            y = model.get_prediction()
            softmax = tf.nn.softmax(y)
            y_p = tf.arg_max(softmax, dimension = 1)
            top_1 = tf.nn.in_top_k(softmax, y_, 1, name = 'top1')
            top_5 = tf.nn.in_top_k(softmax, y_, 5, name = 'top1')

            top_1 = tf.reduce_mean(tf.cast(top_1, tf.float32))
            top_5 = tf.reduce_mean(tf.cast(top_5, tf.float32))

            # correct_prediction = tf.equal(y_, tf.arg_max(softmax, dimension = 1))
            # sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
        
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(model_dir)
                # reader=pywrap_tensorflow.NewCheckpointReader(os.path.join(model_dir,'model.ckpt-1001'))
                # var_to_shape_map=reader.get_variable_to_shape_map()
                # for key in var_to_shape_map:
                #     print("tensor_name: ", key)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    top_1_sum = 0
                    top_5_sum = 0
                    for i in range(17):
                        xs, ys = next_element
                        ys = tf.reshape(ys,[473])
                        x_input, y_input = sess.run([xs,ys])
                        y_input -= 1
                        top_1_, top_5_ , y_pred= sess.run([top_1,top_5,y_p],feed_dict={x: x_input, y_: y_input})
                        top_1_sum += top_1_
                        top_5_sum += top_5_
                        if k == 0:
                            y_true += list(y_input)
                            y_predict += list(y_pred)


                    top_1_score = top_1_sum / 17 
                    rate_top_1_save.append(top_1_score)
                    top_5_score = top_5_sum / 17
                    rate_top_5_save.append(top_5_score)                    
                    k += 1
                else:
                    print('No checkpoint file found')
                    return 

    top_1_mean = np.mean(rate_top_1_save)
    top_5_mean = np.mean(rate_top_5_save)
    print('{0} Top1 accuracy score in test set is: {1}'.format(NET_TYPE, top_1_mean))
    print('{0} Top5 accuracy score in test set is: {1}'.format(NET_TYPE, top_5_mean))
    
    matrix = confusion_matrix(y_true,y_predict, labels=list(range(0,196)))
    np.save('{0}_matrix.npy'.format(NET_TYPE), matrix)
    # plt.figure(figsize=(16,6))
    # plt.rcParams['font.family'] = 'SimSun'
    # plt.title('{0} 测试集混淆矩阵'.format(NET_TYPE))
    # sns.heatmap(matrix,linewidths=0,cmap="YlGnBu",fmt="d",annot=True)
    # plt.savefig('{0} 测试集混淆矩阵.png'.format(NET_TYPE), bbox_inches='tight')

def main(argv=None):
    train_list = None
    if NET_TYPE == 'alexnet':
        print('--select AlexNet--')
        net = AlexNet
        if TRAIN_MODEL == 'parttune':
            train_list = ['fc8', 'fc6', 'fc5']
    elif NET_TYPE == 'googlenet':
        print('--select GoogLeNet--')
        net = GoogLeNet
        if TRAIN_MODEL == 'parttune':
            train_list = ['loss3_classifier', 'inception_5b', 'inception_5a']
    elif NET_TYPE == 'mobilenet':
        print('--select MobileNet--')
        net = MobileNets
        if TRAIN_MODEL == 'parttune':
            train_list = ['Logits', 'Conv2d_13_pointwise', 'Conv2d_13_depthwise']
    elif NET_TYPE == 'mobilenet_0.75':
        print('--select MobileNet_0.75--')
        net = MobileNets_depth
        if TRAIN_MODEL == 'parttune':
            train_list = ['Logits', 'Conv2d_13_pointwise', 'Conv2d_13_depthwise']
    elif NET_TYPE == 'mobilenet_0.5':
        print('--select MobileNet_0.5--')
        net = MobileNets_depth
        if TRAIN_MODEL == 'parttune':
            train_list = ['Logits', 'Conv2d_13_pointwise', 'Conv2d_13_depthwise']
    else:
        raise ValueError('net type enter error, please input writer error')
    
    evaluate(net, train_list)

if __name__ =='__main__':
    tf.app.run()