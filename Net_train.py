import tensorflow as tf
import numpy as np

from AlexNet import AlexNet
from GoogLeNet import GoogLeNet
from MobileNets import MobileNets

from train_para import train_para
from cifar_data_read import cifar

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('net_model', choices = ['alexnet', 'googlenet','mobilenet'], default='folder', help='choose net')
parser.add_argument('train_model', choices = ['finetune', 'fulltrain','parttune'], default='folder', help='choose net')
parser.add_ar
parser.add_argument('--data_dir', type=str, default='', help='input data path')
parser.add_argument('--model_dir', type=str, default='', help='output model path')

FLAGS, _ = parser.parse_known_args()

args = parser.parse_args()
NET_TYPE = args.net_type
TRAIN_MODEL = args.train_model
DATA_PATH = args.data_dir
MODEL_PATH = args.model_dir

MODEL_NAME = 'model.ckpt'

'''AlexNet'''
AlexNet_fine_tune_para = train_para(
    image_size = 227, lr = 0.001, lr_decay = 0.1,
    train_steps = 100000, train_type = 'fine tune'
)

AlexNet_part_tune_para = train_para(
    image_size = 227, lr = 0.001, lr_decay = 0.1,
    train_steps = 80000, train_type = 'part tune'
)

AlexNet_full_train_para = train_para(
    image_size = 227, lr = 0.001, lr_decay = 0.1,
    train_steps = 100000, train_type = 'full train'
)

'''GoogLeNet V1'''
GoogLeNet_fine_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'fine tune'
)

GoogLeNet_part_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'part tune'
)

GoogLeNet_full_train_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'full train'
)

'''MobileNet V1 1.0 224'''
MobileNet_fine_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'fine tune'
)

MobileNet_part_tune_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'part tune'
)

MobileNet_full_train_para = train_para(
    image_size = 224, lr = 0.0001, lr_decay = 0.96,
    train_steps = 50000, train_type = 'full train'
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
        'fulltrain': GoogLeNet_full_trian_para, 
    },
    'moblienet':{
        'finetune': MobileNet_fine_tune_para,
        'parttune': MobileNet_part_tune_para,
        'fulltrain': MobileNet_full_trian_para,
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
        tf.float32, 
        [None],
        name = 'input-y'
    )

    model = net(x, label, keep_prob, net_para.skip)
    y = model.get_prediction()

    global_step = tf.Variable(0, trainable = False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    one_hot_y = tf.one_hot(y_, 10)
    cross_entropy = one_hot_y*tf.log(y+1e10)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    learning_rate = tf.train.exponential_decay(
        net_para.lr,
        global_step,
        550000 / BATCH_SIZE,
        net_para.lr_decay
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    cif = cifar(DATA_PATH)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for i in range(net_para.train_steps):
            xs, ys = cif.batch(BATCH_SIZE)
            ys = ys.reshape(BATCH_SIZE)
            _, predict, loss_value, step = sess.run([train_op, y, loss, global_step], feed_dict={x: xs, y_: ys})
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
    finally:
        break
    train(net, net_para, )


if __name__ == '__main__':
    tf.app.run()