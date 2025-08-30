# %% 
# TGS Salt Identification Challenge - Segmentation

# 1. IMPORT LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Keras and TensorFlow imports
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import ResNet50, VGG16
from sklearn.model_selection import train_test_split

# For image processing
from skimage.io import imread
from skimage.transform import resize


# 2. LOAD DATA
# Download dataset from Kaggle and set the path
DATA_PATH = "/content/train/"  # Change this if your path is different

# Load image ids
train_images = sorted(glob(os.path.join(DATA_PATH, "images", "*.png")))
train_masks = sorted(glob(os.path.join(DATA_PATH, "masks", "*.png")))

print("Number of images:", len(train_images))
print("Number of masks:", len(train_masks))


# 3. PREPROCESS DATA
IMG_HEIGHT = 128
IMG_WIDTH = 128

def preprocess_image(path):
    img = imread(path, as_gray=True)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img / 255.0  # Normalize
    return img

def preprocess_mask(path):
    mask = imread(path, as_gray=True)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / 255.0  # Normalize
    mask = (mask > 0.5).astype(np.float32)  # Binarize
    return mask

X = np.array([preprocess_image(p) for p in train_images])
Y = np.array([preprocess_mask(p) for p in train_masks])

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


# 4. DEFINE U-NET WITH TRANSFER LEARNING

def build_unet(encoder='resnet', input_shape=(128,128,1)):
    inputs = Input(input_shape)
    
    # Use 3 channels for pretrained models
    x = Conv2D(3, (1,1), padding='same')(inputs)
    
    # Encoder (pretrained)
    if encoder == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
        skip_connection_layers = ["input_1", "conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
    elif encoder == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=x)
        skip_connection_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    else:
        raise ValueError("Encoder not supported")

    # Collect skip connections
    skips = [base_model.get_layer(name).output for name in skip_connection_layers]
    x = base_model.output
    
    # Decoder
    for i in reversed(range(len(skips))):
        x = Conv2DTranspose(256 // (2**i), (2,2), strides=(2,2), padding='same')(x)
        x = concatenate([x, skips[i]])
        x = Conv2D(256 // (2**i), (3,3), activation='relu', padding='same')(x)
        x = Conv2D(256 // (2**i), (3,3), activation='relu', padding='same')(x)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 5. TRAIN AND VALIDATE
def train_model(encoder_name):
    print(f"\nTraining model with {encoder_name} encoder...")
    model = build_unet(encoder=encoder_name)
    
    checkpoint = ModelCheckpoint(f"{encoder_name}_unet.h5", monitor='val_loss', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=8,
        epochs=20,
        callbacks=[checkpoint, earlystop]
    )
    return model, history

# Train with ResNet
resnet_model, resnet_history = train_model('resnet')

# Train with VGG
vgg_model, vgg_history = train_model('vgg')

# 6. EVALUATE RESULTS
def plot_history(history, title):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(resnet_history, 'ResNet U-Net Loss')
plot_history(vgg_history, 'VGG U-Net Loss')

# 7. VISUALIZE PREDICTIONS
def visualize_predictions(model, X, Y, num=3):
    for i in range(num):
        idx = np.random.randint(0, X.shape[0])
        pred = model.predict(np.expand_dims(X[idx], axis=0))[0]
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("Input Image")
        plt.imshow(X[idx].squeeze(), cmap='gray')
        
        plt.subplot(1,3,2)
        plt.title("Ground Truth")
        plt.imshow(Y[idx].squeeze(), cmap='gray')
        
        plt.subplot(1,3,3)
        plt.title("Prediction")
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.show()

print("\nResNet Predictions:")
visualize_predictions(resnet_model, X_val, Y_val)

print("\nVGG Predictions:")
visualize_predictions(vgg_model, X_val, Y_val)
