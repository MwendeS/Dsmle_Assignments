# %% 
# U-NET SEGMENTATION ASSIGNMENT
# This code performs image segmentation using U-Net on the TGS Salt Identification Challenge dataset.

# 1. IMPORT LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 2. DATASET PREPARATION
# Path to your dataset
# Download dataset from Kaggle: https://www.kaggle.com/c/tgs-salt-identification-challenge/data
# train_images_path = 'train/images/'
# train_masks_path = 'train/masks/'

train_images_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\tgs_salt\images"
train_masks_path  = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\tgs_salt\masks"

# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1  # grayscale images

# Load images
def load_dataset(images_path, masks_path=None, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = sorted(glob(os.path.join(images_path, "*.png"))) # accept all
    X = np.zeros((len(images), img_size[0], img_size[1], IMG_CHANNELS), dtype=np.float32)
    
    for i, img_file in enumerate(images):
        img = load_img(img_file, color_mode='grayscale', target_size=img_size)
        X[i] = img_to_array(img) / 255.0  # normalize to 0-1
    
    if masks_path:
        masks = sorted(glob(os.path.join(masks_path, "*.png")))
        Y = np.zeros((len(masks), img_size[0], img_size[1], 1), dtype=np.float32)
        for i, mask_file in enumerate(masks):
            mask = load_img(mask_file, color_mode='grayscale', target_size=img_size)
            Y[i] = img_to_array(mask) / 255.0  # normalize mask
        return X, Y
    
    return X

# Load training data
X_train, Y_train = load_dataset(train_images_path, train_masks_path)

print("Training data shape:", X_train.shape, Y_train.shape)

# 3. BUILD U-NET MODEL
def unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    c4 = Conv2D(128, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3,3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)
    
    # Bottleneck
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation='relu', padding='same')(u8)
    c8 = Conv2D(32, (3,3), activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3,3), activation='relu', padding='same')(u9)
    c9 = Conv2D(16, (3,3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Create the model
model = unet()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. TRAIN THE MODEL
callbacks = [
    EarlyStopping(patience=5, verbose=1, monitor='val_loss'),
    ModelCheckpoint('unet_salt_model.h5', verbose=1, save_best_only=True)
]

#history = model.fit(
#    X_train, Y_train,
#    validation_split=0,
#  batch_size=16,
#  epochs=20,
#  callbacks=callbacks

# Use a smaller subset
X_small = X_train[:50]
Y_small = Y_train[:50]
history = model.fit(
    X_small, Y_small,
    batch_size=8,
    epochs=20

)

# 5. PREDICTION AND VISUALIZATION
# Example prediction
idx = 0
pred_mask = model.predict(np.expand_dims(X_train[idx], axis=0))[0]
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Input Image')
plt.imshow(X_train[idx].squeeze(), cmap='gray')
plt.subplot(1,3,2)
plt.title('Ground Truth Mask')
plt.imshow(Y_train[idx].squeeze(), cmap='gray')
plt.subplot(1,3,3)
plt.title('Predicted Mask')
plt.imshow(pred_mask.squeeze(), cmap='gray')
plt.show()
# %%
