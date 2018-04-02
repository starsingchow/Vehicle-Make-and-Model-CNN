#%%
import tensorflow as tf
from tensorflow.python.framework import graph_util


output_graph_path = '/Users/starsingchow/Downloads/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb'
with tf.Session() as sess:
    # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    output = sess.graph.get_tensor_by_name("MobilenetV1/Logits/Conv2d_1c_1x1/biases:0")
    print(output)


#%%
from tensorflow.python import pywrap_tensorflow
import numpy as np
checkpoint_path='/Users/starsingchow/Downloads/mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt'#your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

mobilenets={}

for key in var_to_shape_map:
    print("tensor_name",key)
    str_name = key
    if str_name.find('RMSProp') > -1:
        continue
    if str_name.find('ExponentialMovingAverage') > -1:
        continue
    if str_name.find('moving_variance') > -1:
        continue 
    if str_name.find('moving_mean') > -1:
        continue
    if str_name.find('global_step') > -1:
        continue

    names = str_name.split('/')
    if mobilenets.get(names[1]) == None:
        mobilenets[names[1]] = {}
    if names[2].find('weights') > -1:
        mobilenets[names[1]][names[2]] = reader.get_tensor(key)
    elif names[2].find('BatchNorm') > -1 :
        if mobilenets[names[1]].get(names[2]) == None:
            mobilenets[names[1]][names[2]] = {}
        mobilenets[names[1]][names[2]][names[3]] = reader.get_tensor(key)
    elif names[2].find('Conv2d_1c_1x1') > -1:
        if mobilenets[names[1]].get(names[2]) == None:
            mobilenets[names[1]][names[2]] = {}
        mobilenets[names[1]][names[2]][names[3]] = reader.get_tensor(key)
    # print("tensor_name",key)
    # sStr_2=key[:-2]
    # print(sStr_2)
    # if mobilenets.get(sStr_2) != None:
    # mobilenets[key]=reader.get_tensor(key)
    # else:
    #     mobilenets[sStr_2].append(reader.get_tensor(key))

np.save('/Users/starsingchow/Downloads/mobilenets_v1_0.5_224.npy',mobilenets)

#%%
import tensorflow as tf
import argparse 

# Pass the filename as an argument
parser = argparse.ArgumentParser()
parser.add_argument("--frozen_model_filename", default="/path-to-pb-file/Binary_Protobuf.pb", type=str, help="Pb model file to import")
args = parser.parse_args()

# We load the protobuf file from the disk and parse it to retrieve the 
# unserialized graph_def
with tf.gfile.GFile(args.frozen_model_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    #saver=tf.train.Saver()
    with tf.Graph().as_default() as graph:

        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
        sess = tf.Session(graph=graph)
        saver=tf.train.Saver()
        save_path = saver.save(sess, "path-to-ckpt/model.ckpt")
        print("Model saved to chkp format")