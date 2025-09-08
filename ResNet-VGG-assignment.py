# %%
# TGS Salt Identification Challenge - Segmentation
# 1. IMPORT LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Resizing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50, VGG16
from sklearn.model_selection import train_test_split

# reproducible-ish
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2. LOAD DATA
DATA_PATH = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\tgs_salt"
train_images = sorted(glob(os.path.join(DATA_PATH, "images", "*.png")))
train_masks  = sorted(glob(os.path.join(DATA_PATH, "masks", "*.png")))

# 3. PREPROCESS DATA
IMG_HEIGHT = 128
IMG_WIDTH  = 128

def preprocess_image(path):
    img = imread(path, as_gray=True)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0  # Normalize
    return img.astype(np.float32)

def preprocess_mask(path):
    mask = imread(path, as_gray=True)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / 255.0  # Normalize
    mask = (mask > 0.5).astype(np.float32)  # Binarize
    return mask.astype(np.float32)

# Limit data to avoid MemoryError
train_images = train_images[:400]
train_masks  = train_masks[:400]

X = np.array([preprocess_image(p) for p in train_images], dtype=np.float32)
Y = np.array([preprocess_mask(p) for p in train_masks], dtype=np.float32)

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# Use a smaller subset for quick testing
X_train = X_train[:2]
Y_train = Y_train[:2]
X_val   = X_val[:2]
Y_val   = Y_val[:2]

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


# 4. DEFINE U-NET WITH TRANSFER LEARNING
def build_unet(encoder='resnet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    inputs = Input(input_shape)

    # Use 3 channels for pretrained models (convert single channel to 3)
    x = Conv2D(3, (1,1), padding='same')(inputs)

    # Encoder
    if encoder == 'resnet':
        base_model = ResNet50(weights=None, include_top=False, input_tensor=x)
        skip_connection_layers = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
    elif encoder == 'vgg':
        base_model = VGG16(weights=None, include_top=False, input_tensor=x)
        skip_connection_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    else:
        raise ValueError("Encoder not supported")

    # Collect skip connections and the bottleneck
    skips = [base_model.get_layer(name).output for name in skip_connection_layers]
    x = base_model.output

    # Decoder (upsample back)
    for i in reversed(range(len(skips))):
        filters = max(16, 256 // (2**i))  # keep filters reasonable
        x = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
        x = concatenate([x, skips[i]])
        x = Conv2D(filters, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3,3), activation='relu', padding='same')(x)

    # ---- EXTRA UPSAMPLE to reach original input resolution (128x128)
    # Without this the output was 64x64; this doubles to 128x128 to match Y masks.
    x = Resizing(IMG_HEIGHT, IMG_WIDTH, interpolation='bilinear')(x)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
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
        batch_size=2,
        epochs=2,
        callbacks=[checkpoint, earlystop]
    )
    return model, history

# %%
# Train with ResNet
resnet_model, resnet_history = train_model('resnet')
# %%
# Train with VGG
vgg_model, vgg_history = train_model('vgg')
# %%
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
# %%
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
# %%
