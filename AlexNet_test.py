import os
import urllib.request
import argparse
import sys
import tensorflow as tf
import numpy as np
import AlexNet as alexnet
import cv2
import caffe_classes

parser = argparse.ArgumentParser(description='Classify some images.')
parser.add_argument('mode', choices = ['folder', 'url'], default='folder', help='data file mode')
parser.add_argument('path', type=str, default='', help='input data folder/path')

args = parser.parse_args(sys.argv[1:])

if args.mode == 'folder':
    withPath = lambda file: '{}/{}'.format(args.path,file)
    testImg = dict((file, cv2.imread(withPath(file))) for file in os.listdir(args.path) if os.path.isfile(withPath(file)))
elif args.mode == 'url':
    def url2img(url):
        '''url to image'''
        resp = urllib.request.urloepn(url)
        image = np.asarray(bytearray(resp.read()), dtype = 'uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    testImg = {args.path:url2img(args.path)}

if testImg.values():
    #params
    keep_drop = 1
    num_label = 1000
    skip = []

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder('float', [1, 227, 227, 3])

    model = alexnet.AlexNet(x, num_label, keep_drop, skip)
    score = model.fc8
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for key,img in testImg.items():
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
            print("{}: {}\n----".format(key, res))
            cv2.imshow("demo", img)
            cv2.waitKey(0)