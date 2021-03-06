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
    images = dict((file, [cv2.imread(withPath(file)), car_data.loc[car_data['relative_im_path'] == file, 
            ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]])
            for file in files if os.path.isfile(withPath(file)) and file != '.DS_Store')
    test_dir = os.path.join(save, 'test')
    if not os.path.exists(test_dir):
        print('--create test file--')
        os.makedirs(test_dir)
                
    train_dir = os.path.join(save, 'train')
    if not os.path.exists(train_dir):
        print('--create train file--')
        os.makedirs(train_dir)
    
    valid_dir = os.path.join(save, 'valid')
    if not os.path.exists(valid_dir):
        print('--create valid file--')
        os.makedirs(valid_dir)

    for key, image in images.items():
        rgb_image = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
        bbox = image[1]
        bbox_image = bbox_crop(rgb_image, bbox)
        # print(bbox_image.shape)
        squash_image = Squash(bbox_image)

        name_label = car_data.loc[car_data['relative_im_path'] == key, ['class', 'test']]
        if int(name_label['test']) == 1:
            cropped = RandomCrop(squash_image, 1)
            save_dir = test_dir
            file_name = 'num_1_label_{0}_'.format(int(name_label['class'])) + key
            save_name = os.path.join(save_dir, file_name)
            cv2.imwrite(save_name, cropped[0])
            # print(save_name)
            # print('finsh test {0}'.format(key))
            continue

        # print(squash_image.shape)
        # cv2.imshow('a', cv2.cvtColor(squash_image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
        # squash_image =Squash(rgb_image)
        flipped_image = RandomFlip(squash_image)
            
        results = [squash_image] + [flipped_image]

        ratio_results = []
        for result in results:
            ratio_images = RandomRatio(result, 1)
            ratio_results += ratio_images
        results +=  ratio_results

        # print('finsh ratio')
        # brightness_results = []
        # for result in results:
        #     brightness = Brightness(result, 2)
        #     brightness_results += brightness
        # results += brightness_results

        # print('finsh brightness')

        # colorJittering_results = []
        # for result in results:
        #     colorJittering = ColorJittering(result, 3)
        #     colorJittering_results += colorJittering
        # results += colorJittering_results
        # print('finsh colorJittering')

        cropped_images = []
        for result in results:
            cropped = RandomCrop(result, 2)
            cropped_images += cropped
        random.seed(2233)
        # print('finsh crop')
        try:
            save_number = random.sample(range(2,len(cropped_images)), 5)
            valid = save_number[4]
            save_number = save_number[0:4]
            EdgeEnhance_number = random.sample(range(2,len(cropped_images)),2)
            colorJittering_number = random.sample(range(1,len(cropped_images),2), 2)
            brightness_number = random.sample(range(2,len(cropped_images),2), 2) 
        except ValueError:
            save_number = []
            valid = []
            EdgeEnhance_number = random.sample(range(len(cropped_images)), 1)
            colorJittering_number = random.sample(range(len(cropped_images)), 1)
            brightness_number = random.sample(range(len(cropped_images)), 1) 
            
        i = 0
        for cropped_image in cropped_images:
            if i not in save_number and i != 0 and i != valid:
                i +=1
                continue
            if i in brightness_number:
                cropped_image = Brightness(cropped_image)

            if i in colorJittering_number:
                cropped_image = ColorJittering(cropped_image)

            if i in EdgeEnhance_number and i not in colorJittering_number:
                    cropped_image = EdgeEnhance(cropped_image)

            save_dir= train_dir
            if i == valid:
                save_dir = valid_dir

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            save_name = save_dir + '/num_{0}_label_{1}_'.format(i,int(name_label['class'])) + key
            # print(save_name)
            cv2.imwrite(save_name, cropped_image)
            i += 1
        # print('finsh train {0}'.format(key))
    
    print('finish data augmentation')
            
def bbox_crop(image, bbox_crop):
    image_shape = image.shape
    # if image_shape[0] < 224 and image_shape[1] < 224:
    #     return image
    # elif image_shape[1] < 224:
    #     y1 = bbox_crop['bbox_y1'].get_values()[0] - 16
    #     if y1 < 0:
    #         y1 = 0

    #     y2 = bbox_crop['bbox_y2'].get_values()[0] + 16
    #     if y2 > image_shape[0]:
    #         y2 = image_shape[0]
    #     return image[y1:y2, :]
    # elif image_shape[0] < 224:
    #     x1 = bbox_crop['bbox_x1'].get_values()[0] - 16
    #     if x1 < 0:
    #         x1 = 0

    #     x2 = bbox_crop['bbox_x2'].get_values()[0] + 16
    #     if x2 > image_shape[1]:
    #         x2 = image_shape[1]
    #     return image[:, x1:x2]
    # else:
    #     x1 = bbox_crop['bbox_x1'].get_values()[0] - 16
    #     if x1 < 0:
    #         x1 = 0

    #     x2 = bbox_crop['bbox_x2'].get_values()[0] + 16
    #     if x2 > image_shape[1]:
    #         x2 = image_shape[1]

    #     y1 = bbox_crop['bbox_y1'].get_values()[0] - 16
    #     if y1 < 0:
    #         y1 = 0

    #     y2 = bbox_crop['bbox_y2'].get_values()[0] + 16
    #     if y2 > image_shape[0]:
    #         y2 = image_shape[0]
    #     return image[y1:y2, x1:x2, :]

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
    # if image_shape[0] < 256 and image_shape[1] < 256:
    #     return image
    
    # if image_shape[0] >=  image_shape[1]:
    #     ratio = 256 / image_shape[0]
    # else:
    #     ratio = 256 /image_shape[1]
    
    image.astype('Float64')
    
    # x = int(image_shape[0] * ratio)
    # y = int(image_shape[1] * ratio)
    tf.reset_default_graph()
    with tf.Session() as sess:
        resized = tf.image.resize_images(image, [256, 256], method = 0)
        eval = sess.run(resized)
    tf.get_default_graph().finalize()
    return eval

def RandomCrop(image, numbers):
    h, w, c = image.shape
    croped_images = []
    tf.reset_default_graph()
    if h <= 227 and w <= 227:
        with tf.Session() as sess:
            pad = tf.image.resize_image_with_crop_or_pad(image, 227, 227)
            eval = sess.run(pad)
        tf.get_default_graph().finalize()
        return eval
    elif h <= 227:
        image = tf.image.resize_image_with_crop_or_pad(image, 227, w)
        numbers = 1
    elif w <= 227:
        image = tf.image.resize_image_with_crop_or_pad(image, h, 227)
        numbers = 1

    for i in range(numbers):
        croped_image = tf.random_crop(image, [227, 227, 3],seed = 345)  
        with tf.Session() as sess:
            crop_eval = sess.run(croped_image)
        croped_images.append(crop_eval)
    tf.get_default_graph().finalize()
    return croped_images

def RandomRatio(image, numbers):
    ratios_images = []
    for i in range(numbers):
        angle = np.random.uniform(low=-10.0, high=10.0)  
        ratios_iamge = misc.imrotate(image, angle, 'bicubic')
        ratios_images.append(ratios_iamge)
    return ratios_images

def RandomFlip(image):
    tf.reset_default_graph()
    with tf.Session() as sess:
        flipped_image = tf.image.flip_left_right(image)
        eval = sess.run(flipped_image)
    tf.get_default_graph().finalize()
    return eval

'''return list'''
# def ColorJittering(image, numbers):
#     h, w, c = image.shape
#     noise_images = []
#     for i in range(numbers):
#         noise = np.random.randint(0, 50, (h, w))
#         zitter = np.zeros_like(image)
#         zitter[:, :, 1] = noise
#         noise_image = cv2.add(image, zitter)
#         noise_images.append(noise_image)
    
#     return noise_images

'''return only one'''
def ColorJittering(image):
    h, w, c = image.shape
    noise = np.random.randint(0, 50, (h, w))
    zitter = np.zeros_like(image)
    zitter[:, :, 1] = noise
    noise_image = cv2.add(image, zitter)
    return noise_image

def EdgeEnhance(image):
    Edge_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return Edge_image

'''return list'''
# def Brightness(image, numbers):
#     brightnessed_images = []
#     for i in range(numbers):
#         brightnessed_image = tf.image.random_brightness(image, 0.5, seed=12345)
#         brightnessed_images.append(brightnessed_image.eval())
#     return brightnessed_images

'''return only one'''
def Brightness(image):
    tf.reset_default_graph()
    with tf.Session() as sess:
        brightnessed_image = tf.image.random_brightness(image, 0.5, seed=2233)
        eval = sess.run(brightnessed_image)
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
