# %%
# Scratch Deep Neural Network Classifier Assignment

import numpy as np
import matplotlib.pyplot as plt
import copy
from tensorflow.keras.datasets import mnist

# Problem 2 & 6: Initializers
class SimpleInitializer:
    """Gaussian initialization with standard deviation sigma"""
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def W(self, n_nodes1, n_nodes2):
        return np.random.randn(n_nodes1, n_nodes2) * self.sigma

    def B(self, n_nodes2):
        return np.zeros((1, n_nodes2))

class XavierInitializer:
    """Xavier initialization for Sigmoid/Tanh"""
    def W(self, n_nodes1, n_nodes2):
        sigma = 1.0 / np.sqrt(n_nodes1)
        return np.random.randn(n_nodes1, n_nodes2) * sigma

    def B(self, n_nodes2):
        return np.zeros((1, n_nodes2))

class HeInitializer:
    """He initialization for ReLU"""
    def W(self, n_nodes1, n_nodes2):
        sigma = np.sqrt(2.0 / n_nodes1)
        return np.random.randn(n_nodes1, n_nodes2) * sigma

    def B(self, n_nodes2):
        return np.zeros((1, n_nodes2))

# Problem 4 & 5: Activation Functions
class Tanh:
    def forward(self, X):
        self.X = X
        return np.tanh(X)

    def backward(self, dA):
        return dA * (1 - np.tanh(self.X) ** 2)

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)

class Softmax:
    def forward(self, X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, Y):
        # cross-entropy derivative simplified with softmax
        return self.out - Y

# Problem 3 & 7: Optimizers
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB
        return layer

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.HW = None
        self.HB = None
        self.eps = 1e-8

    def update(self, layer):
        if self.HW is None:
            self.HW = np.zeros_like(layer.W)
            self.HB = np.zeros_like(layer.B)
        self.HW += layer.dW ** 2
        self.HB += layer.dB ** 2
        layer.W -= self.lr * layer.dW / (np.sqrt(self.HW) + self.eps)
        layer.B -= self.lr * layer.dB / (np.sqrt(self.HB) + self.eps)
        return layer

# Problem 1: Fully Connected Layer
class FC:
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.optimizer = optimizer
        self.W = initializer.W(n_nodes1, n_nodes2)
        self.B = initializer.B(n_nodes2)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.B

    def backward(self, dA):
        self.dW = np.dot(self.X.T, dA) / self.X.shape[0]
        self.dB = np.sum(dA, axis=0, keepdims=True) / self.X.shape[0]
        dZ = np.dot(dA, self.W.T)
        self.optimizer.update(self)
        return dZ

# Problem 8: Scratch Deep Neural Network Classifier
class ScratchDeepNeuralNetrowkClassifier:
    def __init__(self, n_features, n_output, layer_sizes, activations,
                 initializer=SimpleInitializer(), optimizer=SGD(lr=0.01)):
        """
        layer_sizes: list of hidden layer node counts, e.g. [128, 64]
        activations: list of activation class instances, e.g. [Tanh(), ReLU()]
        """
        self.n_features = n_features
        self.n_output = n_output
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.initializer = initializer
        self.optimizer = optimizer
        self.layers = []
        self.activ_funcs = []

        prev_nodes = n_features
        # Hidden layers
        for size, act in zip(layer_sizes, activations):
            # give each FC its own optimizer copy
            self.layers.append(FC(prev_nodes, size, initializer, copy.deepcopy(optimizer)))
            self.activ_funcs.append(act)
            prev_nodes = size

        # Output layer with Softmax
        self.layers.append(FC(prev_nodes, n_output, initializer, copy.deepcopy(optimizer)))
        self.activ_funcs.append(Softmax())

    def forward(self, X):
        out = X
        for layer, act in zip(self.layers, self.activ_funcs):
            out = act.forward(layer.forward(out))
        return out

    def backward(self, Y):
        # First step: gradient from softmax cross-entropy
        dA = self.activ_funcs[-1].backward(Y)
        # Backprop through the output layer (no extra activation here)
        dA = self.layers[-1].backward(dA)
        # Backprop through hidden layers with their activations
        for layer, act in reversed(list(zip(self.layers[:-1], self.activ_funcs[:-1]))):
            dA = act.backward(dA)
            dA = layer.backward(dA)

    # REQUIRED 1: fit with mini-batch & learning curve
    def fit(self, X, Y, epochs=10, batch_size=None, val_data=None):
        n_samples = X.shape[0]
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Shuffle the data
            idx = np.random.permutation(n_samples)
            X, Y = X[idx], Y[idx]

            if batch_size is None:
                batch_size = n_samples  # use full batch if batch_size not set

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch, Y_batch = X[start:end], Y[start:end]
                out = self.forward(X_batch)
                self.backward(Y_batch)

            # Compute train loss
            Y_pred = self.forward(X)
            loss = -np.mean(np.sum(Y * np.log(Y_pred + 1e-8), axis=1))
            history['loss'].append(loss)

            # Compute validation loss if validation data is provided
            if val_data:
                X_val, Y_val = val_data
                Y_val_pred = self.forward(X_val)
                val_loss = -np.mean(np.sum(Y_val * np.log(Y_val_pred + 1e-8), axis=1))
                history['val_loss'].append(val_loss)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, loss={loss:.4f}", end='')
            if val_data:
                print(f", val_loss={val_loss:.4f}")
            else:
                print()

        # Plot learning curve
        plt.plot(history['loss'], label='Train Loss')
        if val_data: plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()
    # ----------------------------------------

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)

    def accuracy(self, X, Y_true):
        Y_pred = self.predict(X)
        Y_true_idx = np.argmax(Y_true, axis=1)
        return np.mean(Y_pred == Y_true_idx)
# %%
# REQUIRED 2: MNIST images
if __name__ == "__main__":
    # Load MNIST
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
    X_train_mnist = X_train_mnist / 255.0
    X_test_mnist = X_test_mnist / 255.0

    # Flatten
    X_train_flat = X_train_mnist.reshape(-1, 28*28)
    X_test_flat = X_test_mnist.reshape(-1, 28*28)

    # One-hot encode labels
    Y_train_mnist = np.eye(10)[y_train_mnist]
    Y_test_mnist = np.eye(10)[y_test_mnist]

    # Display 5 MNIST images
    for i in range(5):
        plt.imshow(X_train_mnist[i], cmap='gray')
        plt.title(f"Label: {y_train_mnist[i]}")
        plt.axis('off')
        plt.show()

    # Create model
    model = ScratchDeepNeuralNetrowkClassifier(
        n_features=784,
        n_output=10,
        layer_sizes=[128, 64],
        activations=[Tanh(), ReLU()],
        initializer=XavierInitializer(),
        optimizer=SGD(lr=0.01)
    )

    # Train model
    model.fit(
        X_train_flat,
        Y_train_mnist,
        epochs=20,
        batch_size=64,
        val_data=(X_test_flat, Y_test_mnist)
    )

    # Print accuracies
    print("Training accuracy:", model.accuracy(X_train_flat, Y_train_mnist))
    print("Validation accuracy:", model.accuracy(X_test_flat, Y_test_mnist))

# %%
