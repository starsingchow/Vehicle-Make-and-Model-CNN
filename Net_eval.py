import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
import numpy as np

from AlexNet import AlexNet
from GoogLeNet import GoogLeNet
from MobileNets import MobileNets

from input_data import get_test_data

import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

parser = argparse.ArgumentParser()
parser.add_argument('net_model', choices = ['alexnet', 'googlenet','mobilenet'], default='folder', help='choose net')
parser.add_argument('train_model', choices = ['finetune', 'fulltrain','parttune'], default='folder', help='choose net')
parser.add_argument('--interval_secs', type=str, default='', help ='eval waiting times')
parser.add_argument('--data_dir', type=str, default='', help ='Valid Data Dir')
parser.add_argument('--model_dir', type=str, default='', help ='Model Dir')
parser.add_argument('--log_dir', type=str, default='', help ='Log Dir')

FLAGS, _ = parser.parse_known_args()

args = parser.parse_args()
NET_TYPE = args.net_model
TRAIN_MODEL = args.train_model
EVAL_INTERVAL_SECS = args.interval_secs
DATA_PATH = args.data_dir
MODEL_DIR = args.model_dir
LOG_DIR = args.log_dir
NUMBER_CHANNEL = 3
IMAGE_SIZE = 224
MOVING_AVERAGE_DECAY = 0.99

if NET_TYPE == 'alexnet':
    IMAGE_SIZE = 227
model_dir = os.path.join(MODEL_DIR, NET_TYPE, TRAIN_MODEL)

def evaluate(net,log_dir, trian_list):
    with tf.Graph().as_default() as g:
        data_iterator = get_test_data(DATA_PATH, 1018)
        next_element = data_iterator.get_next()
        x = tf.placeholder(
            tf.float32,
            [1018, IMAGE_SIZE, 
            IMAGE_SIZE, NUMBER_CHANNEL],
            name = 'input-x'
            )
        y_ = tf.placeholder(
            tf.int64, 
            [None],
            name = 'input-y'
        )

        model = net(x, 196, 1, None, train_list=trian_list)
        y = model.get_prediction()
        softmax = tf.nn.softmax(y)

        correct_prediction = tf.equal(y_, tf.arg_max(softmax, dimension = 1))
        sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        rate_set = {}
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(model_dir)
                # reader=pywrap_tensorflow.NewCheckpointReader(os.path.join(model_dir,'model.ckpt-1001'))
                # var_to_shape_map=reader.get_variable_to_shape_map()
                # for key in var_to_shape_map:
                #     print("tensor_name: ", key)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]
                    all_sum = 0
                    for i in range(16):
                        xs, ys = next_element
                        ys = tf.reshape(ys,[509])
                        x_input, y_input = sess.run([xs,ys])
                        y_input -= 1
                        sum_ = sess.run([sum],feed_dict={x: x_input, y_: y_input})
                        all_sum += sum_[0]

                    accuracy_score = all_sum / 8144
                    print('After {0:s} training step(s), validation accuracy {1:g}'.format(global_step, accuracy_score))
                    rate_set[str(global_step)] = accuracy_score
                else:
                    print('No checkpoint file found')
                    return
            
            with open(log_dir+'/rate_set.txt', 'w') as f:
                f.write(str(rate_set))      
            
        time.sleep(EVAL_INTERVAL_SECS)

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
    else:
        raise ValueError('net type enter error, please input writer error')
    
    log_dir = os.path.join(LOG_DIR, NET_TYPE, TRAIN_MODEL, 'valid')
    if not os.path.exists(log_dir):
        print('--create log file--')
        os.makedirs(log_dir)
    evaluate(net,log_dir, train_list)

if __name__ =='__main__':
    tf.app.run()
                    





