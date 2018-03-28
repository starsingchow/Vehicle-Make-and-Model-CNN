import tensorflow as tf
import numpy as np

from AlexNet import AlexNet
from GoogLeNet import GoogLeNet
from MobileNets import MobileNets

from input_data import get_test_data

import argparse
import os
import time

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

def evaluate(net,log_dir):
    with tf.Graph().as_default() as g:
        data_iterator = get_test_data(DATA_PATH, 8144)
        next_element = data_iterator.get_next()
        x, y_ = next_element
        x = tf.cast(x, tf.float32)
        y_ = tf.cast(y_, tf.int64)
        y_ = tf.reshape(y_,[8144])

        model = net(x, 196, 1, None)
        y = model.get_prediction()
        softmax = tf.nn.softmax(y)

        correct_prediction = tf.equal(y_, tf.arg_max(softmax, dimension = 1))
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        merged = tf.summary.merge_all()

        while True:
            with tf.Session() as sess:
                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                ckpt = tf.train.get_checkpoint_state(model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]
                    accuracy_score,summary= sess.run([accuracy,merged])
                    print('After {0:s} training step(s), validation accuracy {1:g}'.format(global_step, accuracy_score))
                    summary_writer.add_summary(summary,global_step)
                else:
                    print('No checkpoint file found')
                    return
                
                summary_writer.close()


        time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    if NET_TYPE == 'alexnet':
        print('--select AlexNet--')
        net = AlexNet
    elif NET_TYPE == 'googlenet':
        print('--select GoogLeNet--')
        net = GoogLeNet
    elif NET_TYPE == 'mobilenet':
        print('--select MobileNet--')
        net = MobileNets
    else:
        raise ValueError('net type enter error, please input writer error')
    
    log_dir = os.path.join(LOG_DIR, NET_TYPE, TRAIN_MODEL, 'valid')
    if not os.path.exists(log_dir):
        print('--create log file--')
        os.makedirs(log_dir)
    evaluate(net,log_dir)

if __name__ =='__main__':
    tf.app.run()
                    





