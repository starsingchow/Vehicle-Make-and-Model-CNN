import os
import urllib.request
import argparse
import sys
import tensorflow as tf
import numpy as np
import pandas
import AlexNet as alexnet
import GoogLeNet as googlenet
import MobileNets as mobilenets
import cv2


parser = argparse.ArgumentParser(description='Classify some images.')
parser.add_argument('mode', choices = ['folder', 'url'], default='folder', help='data file mode')
parser.add_argument('net_model', choices = ['alexnet', 'googlenet','mobilenet'], default='googlenet', help='choose net')
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
    keep_drop = 0.4
    num_label = 176
    skip = []

    x = tf.placeholder('float', [1, 227, 227, 3])

    model = alexnet.AlexNet(x, num_label, keep_drop, skip)
    score = model.get_prediction()



    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('./CNN/log', sess.graph)
        sess.run(tf.global_variables_initializer())
        summary_writer.close()

        model.loadModel(sess)

        for key,img in testImg.items():
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(score, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
            print("{}: {}\n----".format(key, res))
            cv2.imshow("demo", img)
            cv2.waitKey(0)

