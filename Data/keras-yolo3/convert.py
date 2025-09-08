# coding: utf-8
# Converts YOLOv3 Darknet weights to TensorFlow checkpoint

import os
import tensorflow as tf
import numpy as np

from yolo3.model import YOLOv3
from utils.misc_utils import parse_anchors, load_weights

# ====================== PATHS ======================
# Darknet weights file (YOLOv3 or YOLOv3-tiny)
weight_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\keras-yolo3\model_data\yolov3-tiny.weights"

# YOLO anchors text file
anchors_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\keras-yolo3\model_data\yolo_anchors.txt"

# Output TensorFlow checkpoint
save_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\keras-yolo3\model_data\yolov3-tiny.ckpt"

# ====================== CONFIG ======================
num_class = 80         # Change if you have a custom dataset
img_size = 416         # Input image size

# ====================== MODEL ======================
anchors = parse_anchors(anchors_path)
model = YOLOv3(num_class, anchors)

# ====================== TF 1.x SESSION ======================
tf.compat.v1.disable_eager_execution()  # Ensure TF1 behavior
with tf.compat.v1.Session() as sess:
    inputs = tf.compat.v1.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.compat.v1.variable_scope('yolov3'):
        feature_maps = model.forward(inputs)

    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(scope='yolov3'))

    # Load Darknet weights
    load_ops = load_weights(tf.compat.v1.global_variables(scope='yolov3'), weight_path)
    sess.run(load_ops)

    # Save checkpoint
    saver.save(sess, save_path)
    print("TensorFlow checkpoint saved to:", save_path)
