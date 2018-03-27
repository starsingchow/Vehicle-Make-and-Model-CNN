import tensorflow as tf
import numpy as np
import cv2

import os
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--dir', type=str, default='', help='dir')
# parser.add_argument('--batch', type=int, default='', help='batch')
# FLAGS, _ = parser.parse_known_args()
# args = parser.parse_args()

# DIR  = args.dir
# BATCH_SIZE = args.batch

# def _read_py_function(filename, label):
#     print(filename)
#     # file = os.path.join(DIR, filename)
#     # image = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_string = tf.read_file(filename)
#     image = tf.image.decode_image(image_string)
#     return image, label

def get_label(filename):
    # filename: num_x_label_x_xxxxx.jpg
    name_split = filename.split('_')
    label = name_split[3]
    return int(label)

# def get_data(dir, BATCH_SIZE):
#     withPath = lambda file: '{}/{}'.format(dir,file)
#     filenames = os.listdir(dir)
#     filenames = [file for file in filenames if os.path.isfile(withPath(file)) and file != '.DS_Store' ]
#     labels = [get_label(file) for file in filenames]
#     filenames = [withPath(file) for file in filenames]
#     filenames = np.array(filenames).reshape(1, len(filenames))
#     labels = np.array(labels).reshape(1, len(labels))
#     filenames = tf.constant(filenames)
#     print(filenames)
#     labels = tf.constant(labels)

#     data = tf.data.Dataset.from_tensor_slices((filenames, labels))
#     print(data)
#     # read = lambda filename, label: tf.py_func(_read_py_function, 
#     #                 [filename, label],[tf.float32,tf.int32])
#     # print(read)
#     data = data.map(lambda filename, label: tf.py_func(_read_py_function, 
#                     [filename, label],[tf.float32,tf.int32]))

#     data = data.map(_read_py_function)

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(10):
#             dataset = data.shuffle(buffer_size=1000).batch(BATCH_SIZE).repeat(3)
#             # a = sess.run(dataset)
#             print(dataset)


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

def get_data(dir,batch_size):
    list_names = os.listdir(dir)
    filenames = [os.path.join(dir,file) for file in list_names 
                if os.path.isfile(os.path.join(dir,file)) and file != '.DS_Store']
    filenames = tf.constant(filenames)

    labels_list = [get_label(file) for file in list_names if file != '.DS_Store']
    labels_list = tf.constant(labels_list)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels_list))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000).repeat(205).batch(100)

    iterator = dataset.make_one_shot_iterator()
    # images, labels = iterator.get_next()
    return iterator
        


# def main(argv=None):
#     get_data(DIR, BATCH_SIZE)

# if __name__=='__main__':
#     tf.app.run()
