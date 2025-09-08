# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# tf.config.run_functions_eagerly(True) # force execution at the start
tf.random.set_seed(42)
np.random.seed(42)

def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60 + "\n")

# -------------------------
# Problem 1 - MNIST Dense
# -------------------------
def problem1_mnist_official_style(epochs=3):
    print_header("Problem 1 — MNIST Dense")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28*28)).astype("float32") / 255.0
    x_test  = x_test.reshape((-1, 28*28)).astype("float32") / 255.0

    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(28*28,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("Training MNIST dense model...")
    # USE NUMPY ARRAYS — no tf.data.Dataset
    model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=128, verbose=2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"MNIST test accuracy: {acc:.4f}, test loss: {loss:.4f}")

# -------------------------
# Problem 3 - Iris binary
# -------------------------
def problem3_iris_binary(epochs=20):
    print_header("Problem 3 — Iris binary")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    mask = (y==1) | (y==2)
    X, y = X[mask], (y[mask]==2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Training Iris binary model...")
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Iris binary test accuracy: {acc:.4f}, test loss: {loss:.4f}")

# -------------------------
# Problem 4 - Iris multi-class
# -------------------------
def problem4_iris_multiclass(epochs=30):
    print_header("Problem 4 — Iris multi-class")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("Training Iris multi-class model...")
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Iris multi-class test accuracy: {acc:.4f}, test loss: {loss:.4f}")

# -------------------------
# Problem 5 - House Prices regression
# -------------------------
def problem5_house_prices_regression(epochs=5):
    print_header("Problem 5 — House Prices regression")
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print("Training regression model...")
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=1)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Regression test MSE: {loss:.4f}, MAE: {mae:.4f}")

# -------------------------
# Problem 6 - MNIST CNN
# -------------------------
def problem6_mnist_cnn(epochs=5):
    print_header("Problem 6 — MNIST CNN")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
    x_test  = np.expand_dims(x_test.astype("float32") / 255.0, -1)

    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("Training MNIST CNN model...")
    model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=128, verbose=2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"MNIST CNN test accuracy: {acc:.4f}, test loss: {loss:.4f}")

# -------------------------
if __name__ == "__main__":
    problem1_mnist_official_style(epochs=3)
    problem3_iris_binary(epochs=30)
    problem4_iris_multiclass(epochs=30)
    problem5_house_prices_regression(epochs=30)
    problem6_mnist_cnn(epochs=3)

    print("\nAll done! This WILL run cleanly on Windows/VS Code.")
# %%
