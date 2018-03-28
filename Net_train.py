
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
parser.add_argument('--label', type=int, default='', help ='input label number')
parser.add_argument('--car_data', type=str, default='', help='input data path')
parser.add_argument('--model_dir', type=str, default='', help='output model path')
parser.add_argument('--log_dir', type=str, default='', help ='log path')
FLAGS, _ = parser.parse_known_args()

args = parser.parse_args()
NET_TYPE = args.net_model
LABEL = args.label
TRAIN_MODEL = args.train_model
DATA_PATH = args.car_data
MODEL_PATH = args.model_dir
LOG_DIR = args.log_dir

# MEAN_VALUE = 'mean224.npy'
# if NET_TYPE == 'alexnet':
#     MEAN_VALUE = 'mean227.npy'

MODEL_NAME = 'model.ckpt'

'''AlexNet'''
AlexNet_fine_tune_para = train_para(
    image_size = 227, lr = 0.01, lr_decay = 0.9,
    train_steps = 100000, train_type = 'fine tune',
    skip = ['fc8'],train_list = ['fc8']
)

AlexNet_part_tune_para = train_para(
    image_size = 227, lr = 0.01, lr_decay = 0.9,
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
    image_size = 224, lr = 0.00001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'fine tune',
    skip = ['loss3_classifier'], train_list = ['loss3_classifier']
)

GoogLeNet_part_tune_para = train_para(
    image_size = 224, lr = 0.00001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'part tune',
    skip = ['loss3_classifier'], train_list = ['loss3_classifier', 'inception_5b', 'inception_5a']
)

GoogLeNet_full_train_para = train_para(
    image_size = 224, lr = 0.00001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'full train',
    skip = ['loss3_classifier'],
)

'''MobileNet V1 1.0 224'''
MobileNet_fine_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'fine tune',
    skip = ['Logits'],train_list = ['Logits']
)

MobileNet_part_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'part tune',
    skip = ['Logits'],
    train_list = ['Logits', 'Conv2d_13_pointwise', 'Conv2d_13_depthwise']
)

MobileNet_full_train_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'full train',
    skip = ['Logits']
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
BATCH_SIZE = 512
NUMBER_CHANNEL = 3
MOVING_AVERAGE_DECAY = 0.99

def train(net, net_para, label, keep_prob, save_dir):
    data_iterator = get_data(DATA_PATH, BATCH_SIZE)
    next_element = data_iterator.get_next()
    # x_mean = np.load('./Vehicle-Make-and-Model-CNN/data/'+MEAN_VALUE)
    x = tf.placeholder(
        tf.float32,
        [BATCH_SIZE, net_para.image_size, 
        net_para.image_size, NUMBER_CHANNEL],
        name = 'input-x'
        )
    y_ = tf.placeholder(
        tf.int64, 
        [None],
        name = 'input-y'
    )

    model = net(x, label, keep_prob, net_para.skip, train_list=net_para.train_list)
    y = model.get_prediction()
    
    global_step = tf.Variable(0, trainable = False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    # one_hot_y = tf.one_hot(y_, LABEL)
    # cross_entropy = one_hot_y*tf.log(y+1e10)

    # cross_entropy_mean = tf.reduce_mean(cross_entropy)
    logits =  y+1e10
    cross_entropy_mean = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)) / BATCH_SIZE
    with tf.name_scope('loss'):
        loss = cross_entropy_mean
        tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,axis=1),y_)
        correct_rate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', correct_rate)

    learning_rate = tf.train.exponential_decay(
        net_para.lr,
        global_step,
        net_para.train_steps / BATCH_SIZE,
        net_para.lr_decay
    )

    if isinstance(model, MobileNets) and (TRAIN_MODEL == 'finetune' or TRAIN_MODEL == 'parttune'):
        # train_step = tf.train.RMSPropOptimizer(net_para.lr, net_para.lr_decay).minimize(loss, global_step=global_step, 
        #                                     var_list = tf.get_collection('train'))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                var_list = tf.get_collection('train'))
    else:
        # train_step = tf.train.RMSPropOptimizer(net_para.lr, net_para.lr_decay).minimize(loss, global_step=global_step)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for i in range(net_para.train_steps):
            xs, ys = next_element
            ys = tf.reshape(ys,[BATCH_SIZE])
            x_input, y_input = sess.run([xs,ys])
            # x_input -= x_mean
            y_input -= 1
            _, rate, loss_value, step, summary = sess.run([train_op, correct_rate, loss, global_step, merged], feed_dict={x: x_input, y_: y_input})
            
            # print('-- origin --')
            # print(y_input)
            # print('-- predict --')
            # print(np.argmax(y_pred,axis=1))
            # print('-- compare--')
            # print(np.equal(y_input, np.argmax(y_pred,axis=1)))
            # print('-- rate --')
            # pre_rate = np.sum(np.equal(y_input, np.argmax(y_pred,axis=1))) / BATCH_SIZE
            # print(pre_rate)
            # print('-- loss --')
            # print('loss is {0}'.format(loss_value))

            if i % 10 == 0:
                print("After {0:d} training step(s), loss on trian batch {1:g}".format(step, loss_value))
                print("After {0:d} training step(s), correct rate on trian batch {1:s}".format(step, str(rate.astype(np.float))))
            summary_writer.add_summary(summary,i)

        summary_writer.close()
        saver.save(sess, os.path.join(save_dir, MODEL_NAME), global_step=global_step)

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
    
    try:
        net_para = net_paras[NET_TYPE][TRAIN_MODEL]
        print('-- Net para--')
        print('learning rate: {0}'.format(net_para.lr))
        print('learning rate decay: {0}'.format(net_para.lr_decay))
        print('train steps: {0}'.format(net_para.train_steps))
        print('train type: {0}'.format(net_para.train_type))
    except KeyError as error:
        print('please enter right train type')
        return

    save_model_dir = os.path.join(MODEL_PATH, NET_TYPE, TRAIN_MODEL)
    if not os.path.exists(save_model_dir):
        print('--create save file--')
        os.makedirs(save_model_dir)
    
    log_dir = os.path.join(LOG_DIR, NET_TYPE, TRAIN_MODEL)
    if not os.path.exists(log_dir):
        print('--create log file--')
        os.makedirs(log_dir)

    train(net, net_para, LABEL, 0.5, save_model_dir)


if __name__ == '__main__':
    tf.app.run()
