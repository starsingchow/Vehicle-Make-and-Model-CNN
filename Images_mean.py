import cv2
import numpy as np
import os
import argparse
import sys

parser = argparse.ArgumentParser(description='compute image mean.')
parser.add_argument('--dir', type = str, default='', help = 'input data')
parser.add_argument('--save', type = str, default='', help = 'save data')
args = parser.parse_args(sys.argv[1:])
DIR = args.dir
SAVE = args.save

def image_mean(dir, save):
    list = os.listdir(dir)
    list = [os.path.join(dir, file) for file in list if file != '.DS_Store']
    img_size = 227
    sum_r=0
    sum_g=0
    sum_b=0
    count=0

    for file in list:
        img=cv2.imread(file)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(img_size,img_size))
        sum_r=sum_r+img[:,:,0].mean()
        sum_g=sum_g+img[:,:,1].mean()
        sum_b=sum_b+img[:,:,2].mean()
        count=count+1

    sum_r=sum_r/count
    sum_g=sum_g/count
    sum_b=sum_b/count
    img_mean=[sum_r,sum_g,sum_b]
    np.save(os.path.join(save,'mean.npy'), img_mean)

if __name__ == '__main__':
    image_mean(DIR,SAVE)
