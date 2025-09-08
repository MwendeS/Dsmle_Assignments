
# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
# ================= FASTER R-CNN =================
# faster_rcnn_path = "ObjectDetection"
faster_rcnn_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\ObjectDetection"
os.chdir(faster_rcnn_path)

image_folder = "sample_images"
images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

def show_image(img_path, title="Image"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
show_image(images[0], "Faster R-CNN Example")

input_shape = (224, 224, 3)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in vgg_base.layers:
    layer.trainable = False

x = Flatten()(vgg_base.output)
x = Dense(1024, activation='relu')(x)
x = Dense(2, activation='softmax')(x)  # Example: binary classification
faster_rcnn_model = Model(inputs=vgg_base.input, outputs=x)
faster_rcnn_model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Dummy training to confirm pipeline
y_demo = np.array([[1,0],[0,1]])
X_demo = np.array([cv2.resize(cv2.imread(img), (224,224)) for img in images[:2]])
faster_rcnn_model.fit(X_demo, y_demo, epochs=1, batch_size=1)

# Code reading notes:
# RPN -> model_rpn.py
# RoI Pooling -> roi_helpers.py
# Backbone -> VGG16
# Classifier -> Dense layers on top
# ================= YOLOv3 =================
# yolo_path = "../keras-yolo3"
yolo_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\ObjectDetection\keras-yolo3"
os.chdir(yolo_path)

from yolo3.model import yolo_body
from yolo3.utils import get_classes, get_anchors
from yolo3.yolo import YOLO # replaced with one below
# from yolo import YOLO

# these 2 functions added later after putting yolo3.utils on comment
#def get_classes(classes_path):
#    with open(classes_path, encoding="utf-8") as f:
#        class_names = f.readlines()
#    class_names = [c.strip() for c in class_names if c.strip()]
#    return class_names

#def get_anchors(anchors_path):
#    with open(anchors_path) as f:
#        anchors = f.readline()
#       anchors = [float(x) for x in anchors.split(",")]
#      return np.array(anchors).reshape(-1, 2)
# end of functions

classes_path = "model_data/coco_classes.txt"
anchors_path = "model_data/yolo_anchors.txt"
weights_path = "model_data/yolo.h5"

class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)

yolo = YOLO(model_path=weights_path,
            anchors_path=anchors_path,
            classes_path=classes_path,
            score=0.3,
            iou=0.45,
            model_image_size=(416,416))

def yolo_predict(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, scores, classes_detected = yolo.detect_image(img_rgb)
    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Classes:", classes_detected)
    plt.imshow(img_rgb)
    plt.title("YOLOv3 Detection")
    plt.axis('off')
    plt.show()
yolo_predict("../ObjectDetection/sample_images/image1.jpg")

# Training Simpsons dataset:
# 1. Convert annotations to YOLO format (x_center y_center width height normalized)
# 2. Create train.txt and val.txt
# 3. Run train.py to confirm training:
# python train.py --annotation_file train.txt --classes model_data/simpsons_classes.txt --anchors model_data/yolo_anchors.txt --weights model_data/yolo.h5

# Code reading notes:
# Backbone -> yolo_body()
# Detection layers -> yolo_body output
# Loss -> yolo_loss()
# Preprocessing -> get_random_data()
# Postprocessing -> detect_image()
# %%