# %%
# Scratch Deep Neural Network Classifier Assignment

import numpy as np

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
import copy   # <-- add this at the top of your file

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



    def fit(self, X, Y, epochs=10):
        for _ in range(epochs):
            out = self.forward(X)
            self.backward(Y)

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)

    def accuracy(self, X, Y_true):
        Y_pred = self.predict(X)
        Y_true_idx = np.argmax(Y_true, axis=1)
        return np.mean(Y_pred == Y_true_idx)

# Example Usage (MNIST-like)
if __name__ == "__main__":
    # dummy data for quick testing
    X_train = np.random.rand(100, 784)
    Y_train = np.eye(10)[np.random.randint(0, 10, 100)]

    model = ScratchDeepNeuralNetrowkClassifier(
        n_features=784,
        n_output=10,
        layer_sizes=[128, 64],
        activations=[Tanh(), ReLU()],
        initializer=XavierInitializer(),
        optimizer=AdaGrad(lr=0.01)
    )

    model.fit(X_train, Y_train, epochs=20)
    acc = model.accuracy(X_train, Y_train)
    print("Training accuracy:", acc)
# %%
