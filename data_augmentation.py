import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from scipy import misc

import os
import sys
import random
import argparse

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
    with tf.Session() as sess:
        for key, image in images.items():
            rgb_image = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
            bbox = image[1]
            bbox_image = bbox_crop(rgb_image, bbox)
            # print(bbox_image.shape)
            squash_image = Squash(bbox_image)
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

            print('finsh ratio')
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
            
            print('finsh crop')
            random.seed(12345)
            try:
                save_number = random.sample(range(len(cropped_images)), 6)
                EdgeEnhance_number = random.sample(range(len(cropped_images)), 2)
                colorJittering_number = random.sample(range(len(cropped_images)), 3)
                brightness_number = random.sample(range(len(cropped_images)), 3) 
            except ValueError:
                save_number = []
                EdgeEnhance_number = random.sample(range(len(cropped_images)), 1)
                colorJittering_number = random.sample(range(len(cropped_images)), 1)
                brightness_number = random.sample(range(len(cropped_images)), 1) 
            
            i = 0
            if not os.path.exists(save+'/test'):
                os.makedirs(save+'/test')
            
            if not os.path.exists(save+'/train'):
                os.makedirs(save+'/train')

            for cropped_image in cropped_images:
                if i not in save_number:
                    i +=1
                    continue
                if i in brightness_number:
                    cropped_image = Brightness(cropped_image)

                if i in colorJittering_number:
                    cropped_image = ColorJittering(cropped_image)

                if i in EdgeEnhance_number and i not in colorJittering_number:
                    cropped_image = EdgeEnhance(cropped_image)

                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                name_label = car_data.loc[car_data['relative_im_path'] == key, ['class', 'test']]
                save_type = 'train'
                if int(name_label['test']) == 1:
                    save_type = 'test'
                save_name = save + '{0}/num_{1}_label_{2}_'.format(save_type,i,int(name_label['class'])) + key
                print(save_name)
                cv2.imwrite(save_name, cropped_image)
                i += 1
            
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

    x1 = bbox_crop['bbox_x1'].get_values()[0] - 16
    if x1 < 0:
        x1 = 0

    x2 = bbox_crop['bbox_x2'].get_values()[0] + 16
    if x2 > image_shape[1]:
        x2 = image_shape[1]

    y1 = bbox_crop['bbox_y1'].get_values()[0] - 16
    if y1 < 0:
        y1 = 0

    y2 = bbox_crop['bbox_y2'].get_values()[0] + 16
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
    resized = tf.image.resize_images(image, [256, 256], method = 0)
    return resized.eval()

def RandomCrop(image, numbers):
    h, w, c = image.shape
    croped_images = []
    if h <= 227 and w <= 227:
        pad = tf.image.resize_image_with_crop_or_pad(image, 227, 227)
        return [pad.eval()]
    elif h <= 227:
        image = tf.image.resize_image_with_crop_or_pad(image, 227, w)
        numbers = 1
    elif w <= 227:
        image = tf.image.resize_image_with_crop_or_pad(image, h, 227)
        numbers = 1

    for i in range(numbers):
        croped_image = tf.random_crop(image, [227, 227, 3],seed = 12345)  
        croped_images.append(croped_image.eval())
    return croped_images

def RandomRatio(image, numbers):
    ratios_images = []
    for i in range(numbers):
        angle = np.random.uniform(low=-10.0, high=10.0)  
        ratios_iamge = misc.imrotate(image, angle, 'bicubic')
        ratios_images.append(ratios_iamge)
    return ratios_images

def RandomFlip(image):
    flipped_image = tf.image.flip_left_right(image)
    return flipped_image.eval()

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
    brightnessed_image = tf.image.random_brightness(image, 0.5, seed=12345)
    return brightnessed_image.eval()

if __name__ == '__main__':
    car_data = pd.read_csv('/mnt/sing/Vehicle-Make-and-Model-CNN/car_information.csv')
    data_augmenttation(DIR, car_data, SAVE)
