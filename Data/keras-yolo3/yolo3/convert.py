# coding: utf-8
"""
Convert YOLOv3 Darknet weights to TensorFlow checkpoint (TF 2.x compatible)
"""

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from model import yolov3               # make sure model.py is in the same folder
from utils.misc_utils import parse_anchors, load_weights  # utils folder must be correct

# --- PARAMETERS ---
num_class = 80
img_size = 416

weight_path = './data/darknet_weights/yolov3.weights'  # path to your Darknet weights
anchors_path = './data/yolo_anchors.txt'              # path to anchors file
save_path = './data/darknet_weights/yolov3.ckpt'      # where TF checkpoint will be saved

# --- LOAD ANCHORS ---
anchors = parse_anchors(anchors_path)

# --- CREATE MODEL ---
model = yolov3(num_class, anchors)

with tf.Session() as sess:
    # input placeholder
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    # forward pass
    with tf.variable_scope('yolov3'):
        feature_maps = model.forward(inputs)

    # get all variables in 'yolov3'
    var_list = tf.global_variables(scope='yolov3')

    # load Darknet weights into variables
    load_ops = load_weights(var_list, weight_path)
    sess.run(load_ops)

    # save checkpoint
    saver = tf.train.Saver(var_list=var_list)
    saver.save(sess, save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
