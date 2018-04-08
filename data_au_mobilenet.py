import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from scipy import misc

import os
import sys
import random
import argparse
import datetime

parser = argparse.ArgumentParser(description='Data Augmentation.')
parser.add_argument('--dir', type = str, default='', help = 'input_data')
parser.add_argument('--save_dir', type = str, default='', help ='save path')
args = parser.parse_args(sys.argv[1:])
DIR = args.dir
SAVE = args.save_dir

def data_augmenttation(dir, car_data, save):
    withPath = lambda file: '{}/{}'.format(dir,file)
    files = os.listdir(dir)
    valid_set = random.sample(range(len(files)), int(len(files)*0.4))
    images = dict((file, [cv2.imread(withPath(file)), car_data.loc[car_data['relative_im_path'] == file, 
            ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]])
            for file in files if os.path.isfile(withPath(file)) and file != '.DS_Store')
                
    train_dir = os.path.join(save, 'train')
    if not os.path.exists(train_dir):
        print('--create train file--')
        os.makedirs(train_dir)
    
    valid_dir = os.path.join(save, 'valid')
    if not os.path.exists(valid_dir):
        print('--create valid file--')
        os.makedirs(valid_dir)
    
    i = 0
    for key, image in images.items():
        rgb_image = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
        bbox = image[1]
        bbox_image = bbox_crop(rgb_image, bbox)
        # print(bbox_image.shape)
        flipped_image = RandomFlip(bbox_image)
        all_image = bbox_image + flipped_image

        squash_image = Squash(all_image)

        name_label = car_data.loc[car_data['relative_im_path'] == key, ['class', 'test']]
        if int(name_label['test']) == 1:
            continue

        flipped_image = RandomFlip(bbox_image)
        all_images = [bbox_image] + [flipped_image]
        
        belong_valid = random.sample(range(2),2)

        k = 0
        for all_image in all_images:
            save_dir = train_dir
            if i in valid_set and belong_valid[i] == 1:
                save_dir = valid_dir

            all_image = Squash(all_image)

            all_image = cv2.cvtColor(all_image, cv2.COLOR_RGB2BGR)
            save_name = save_dir + '/num_{0}_label_{1}_'.format(i,int(name_label['class'])) + key
            cv2.imwrite(save_name, all_image)
            k += 1
        i += 1
    
    print('finish data augmentation')
            
def bbox_crop(image, bbox_crop):
    image_shape = image.shape
    x1 = bbox_crop['bbox_x1'].get_values()[0] - 18
    if x1 < 0:
        x1 = 0

    x2 = bbox_crop['bbox_x2'].get_values()[0] + 18
    if x2 > image_shape[1]:
        x2 = image_shape[1]

    y1 = bbox_crop['bbox_y1'].get_values()[0] - 18
    if y1 < 0:
        y1 = 0

    y2 = bbox_crop['bbox_y2'].get_values()[0] + 18
    if y2 > image_shape[0]:
        y2 = image_shape[0]
    return image[y1:y2, x1:x2, :]

def Squash(image):
    image_shape = image.shape
    image.astype('Float64')
    tf.reset_default_graph()
    with tf.Session() as sess:
        resized = tf.image.resize_images(image, [224, 224], method = 0)
        eval = sess.run(resized)
    tf.get_default_graph().finalize()
    return eval

if __name__ == '__main__':
    '''use in ec2'''
    car_data = pd.read_csv('./Vehicle-Make-and-Model-CNN/car_information.csv')
    start = datetime.datetime.now()
    # car_data = pd.read_csv('./CNN/car_information.csv')
    data_augmenttation(DIR, car_data, SAVE)
    end = datetime.datetime.now()
    print(end-start)
