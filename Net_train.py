
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from AlexNet import AlexNet
from GoogLeNet import GoogLeNet
from MobileNets import MobileNets

from train_para import train_para
from input_data import get_data

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('net_model', choices = ['alexnet', 'googlenet','mobilenet'], default='folder', help='choose net')
parser.add_argument('train_model', choices = ['finetune', 'fulltrain','parttune'], default='folder', help='choose net')
parser.add_argument('--label', type=int, default='', help = 'input label number')
parser.add_argument('--car_data', type=str, default='', help='input data path')
parser.add_argument('--model_dir', type=str, default='', help='output model path')

FLAGS, _ = parser.parse_known_args()

args = parser.parse_args()
NET_TYPE = args.net_model
LABEL = args.label
TRAIN_MODEL = args.train_model
DATA_PATH = args.car_dir
MODEL_PATH = args.model_dir

MODEL_NAME = 'model.ckpt'

'''AlexNet'''
AlexNet_fine_tune_para = train_para(
    image_size = 227, lr = 0.001, lr_decay = 0.1,
    train_steps = 100000, train_type = 'fine tune',
    skip = ['fc8'],train_list = ['fc8']
)

AlexNet_part_tune_para = train_para(
    image_size = 227, lr = 0.001, lr_decay = 0.1,
    train_steps = 80000, train_type = 'part tune',
    skip = ['fc8'],train_list = ['fc8', 'fc6', 'fc5']
)

AlexNet_full_train_para = train_para(
    image_size = 227, lr = 0.001, lr_decay = 0.1,
    train_steps = 100000, train_type = 'full train',
    skip = ['fc8']
)

'''GoogLeNet V1'''
GoogLeNet_fine_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'fine tune',
    skip = ['loss3_classifier'], train_list = ['loss3_classifier']
)

GoogLeNet_part_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'part tune',
    skip = ['loss3_classifier'], train_list = ['loss3_classifier', 'inception_5b', 'inception_5a']
)

GoogLeNet_full_train_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'full train',
    skip = ['loss3_classifier'],
)

'''MobileNet V1 1.0 224'''
MobileNet_fine_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'fine tune',
    skip = ['Logits']
)

MobileNet_part_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'part tune',
    skip = ['Logits', 'Conv2d_13_pointwise', 'Conv2d_13_depthwise']
)

MobileNet_full_train_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'full train',
)

net_paras = {
    'alexnet': {
        'finetune': AlexNet_fine_tune_para,
        'parttune': AlexNet_part_tune_para,
        'fulltrain': AlexNet_full_train_para,
    },
    'googlenet':{
        'finetune': GoogLeNet_fine_tune_para,
        'parttune': GoogLeNet_part_tune_para,
        'fulltrain': GoogLeNet_full_train_para, 
    },
    'mobilenet':{
        'finetune': MobileNet_fine_tune_para,
        'parttune': MobileNet_part_tune_para,
        'fulltrain': MobileNet_full_train_para,
    }
}
BATCH_SIZE = 100
NUMBER_CHANNEL = 3
MOVING_AVERAGE_DECAY = 0.99
def train(net, net_para, label, keep_prob):
    x = tf.placeholder(
        tf.float32,
        [BATCH_SIZE, net_para.image_size, 
        net_para.image_size, NUMBER_CHANNEL],
        name = 'input-x'
        )
    y_ = tf.placeholder(
        tf.int32, 
        [None],
        name = 'input-y'
    )

    model = net(x, label, keep_prob, net_para.skip)
    y = model.get_prediction()

    global_step = tf.Variable(0, trainable = False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    one_hot_y = tf.one_hot(y_, 100)
    cross_entropy = one_hot_y*tf.log(y+1e10)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    learning_rate = tf.train.exponential_decay(
        net_para.lr,
        global_step,
        net_para.train_steps / BATCH_SIZE,
        net_para.lr_decay
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)
        data_iterator = get_data(DATA_PATH, 100)

        for i in range(net_para.train_steps):
            xs, ys = data_iterator.get_next()
            ys = tf.reshape(ys,[BATCH_SIZE])
            _, predict, loss_value, step = sess.run([train_op, y, loss, global_step], feed_dict={x: xs.eval(), y_: ys.eval()})
            if i % 100 == 0:
                print("After {0:d} training step(s), loss on trian batch {1:g}".format(step, loss_value))
                correct_rate = np.sum(np.argmax(predict,axis=1) == ys,0)/BATCH_SIZE
                print("After {0:d} training step(s), correct rate on trian batch {1:s}".format(step, str(correct_rate.astype(np.float))))
        
        saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    if NET_TYPE == 'alexnet':
        net = AlexNet
    elif NET_TYPE == 'googlenet':
        net = GoogLeNet
    elif NET_TYPE == 'mobilenet':
        net = MobileNets
    else:
        raise ValueError('net type enter error, please input writer error')
    
    try:
        net_para = net_paras[NET_TYPE][TRAIN_MODEL]
    except KeyError as error:
        print('please enter right train type')
        return

    train(net, net_para, LABEL, 0.5)


if __name__ == '__main__':
    tf.app.run()