# %%
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Problem 3: Optimization Method - SGD
class SGD:
    """Stochastic Gradient Descent"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db
        return layer


# Problem 2 & 6: Initialization Methods
class SimpleInitializer:
    """Gaussian initialization with sigma"""
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def W(self, n_in, n_out):
        # returns shape (n_out, n_in)
        return np.random.randn(n_out, n_in) * self.sigma
    def B(self, n_out):
        return np.zeros(n_out)

class XavierInitializer:
    """Xavier Initialization for tanh/sigmoid"""
    def W(self, n_in, n_out):
        sigma = np.sqrt(1 / n_in)
        return np.random.randn(n_out, n_in) * sigma
    def B(self, n_out):
        return np.zeros(n_out)

class HeInitializer:
    """He Initialization for ReLU"""
    def W(self, n_in, n_out):
        sigma = np.sqrt(2 / n_in)
        return np.random.randn(n_out, n_in) * sigma
    def B(self, n_out):
        return np.zeros(n_out)

# Problem 5: Activation Functions
class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    def backward(self, dA):
        return dA * (self.X > 0)

class Softmax:
    """Softmax activation with cross-entropy loss simplified"""
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.out

    def backward(self, Y):
        # Y is one-hot encoded
        return (self.out - Y) / Y.shape[0]

# Problem 1 & 4: Conv1d Layer
class SimpleConv1d:
    """1D Convolution Layer (1 channel, stride=1, no padding)"""
    def __init__(self, kernel_size, initializer, optimizer):
        self.kernel_size = kernel_size
        self.initializer = initializer
        self.optimizer = optimizer
        # initializer.W(kernel_size, 1) returns shape (1, kernel_size), flatten -> kernel_size
        self.W = initializer.W(kernel_size, 1).flatten()
        self.b = initializer.B(1)

    def forward(self, X):
        self.X = X
        N_in = len(X)
        N_out = N_in - self.kernel_size + 1
        out = np.zeros(N_out)
        for i in range(N_out):
            out[i] = np.sum(X[i:i+self.kernel_size] * self.W) + self.b
        self.N_out = N_out
        return out

    def backward(self, dA):
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dA)
        dX = np.zeros_like(self.X)
        for i in range(self.N_out):
            self.dW += dA[i] * self.X[i:i+self.kernel_size]
            dX[i:i+self.kernel_size] += dA[i] * self.W
        self.optimizer.update(self)
        return dX

# Problem 4: Conv1d Layer with multiple channels
class Conv1d:
    """1D Convolution Layer with multiple input/output channels"""
    def __init__(self, in_channels, out_channels, kernel_size, initializer, optimizer):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.initializer = initializer
        self.optimizer = optimizer
        # initializer.W(in_channels*kernel_size, out_channels) returns shape (out_channels, in_channels*kernel_size)
        # reshape directly to (out_channels, in_channels, kernel_size)
        self.W = initializer.W(in_channels * kernel_size, out_channels).reshape(out_channels, in_channels, kernel_size)
        self.b = initializer.B(out_channels)

    def forward(self, X):
        """
        X shape: (in_channels, N_in)
        """
        self.X = X
        N_in = X.shape[1]
        self.N_out = N_in - self.kernel_size + 1
        out = np.zeros((self.out_channels, self.N_out))
        for o in range(self.out_channels):
            for i in range(self.N_out):
                out[o, i] = np.sum(self.X[:, i:i+self.kernel_size] * self.W[o]) + self.b[o]
        return out

    def backward(self, dA):
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dA, axis=1)
        dX = np.zeros_like(self.X)
        for o in range(self.out_channels):
            for i in range(self.N_out):
                self.dW[o] += dA[o, i] * self.X[:, i:i+self.kernel_size]
                dX[:, i:i+self.kernel_size] += dA[o, i] * self.W[o]
        self.optimizer.update(self)
        return dX

# Problem 2: Fully Connected Layer
class FC:
    def __init__(self, n_in, n_out, initializer, optimizer):
        self.optimizer = optimizer
        # initializer.W returns (n_out, n_in) -> transpose to (n_in, n_out) for X @ W
        self.W = initializer.W(n_in, n_out).T
        self.b = initializer.B(n_out)
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    def backward(self, dA):
        # dA shape: (batch, n_out)
        self.dW = self.X.T @ dA            # (n_in, n_out)
        self.db = np.sum(dA, axis=0)      # (n_out,)
        dX = dA @ self.W.T                # (batch, n_in)
        self.optimizer.update(self)
        return dX

# Problem 8: Scratch1dCNNClassifier
class Scratch1dCNNClassifier:
    def __init__(self, lr=0.01):
        self.optimizer = SGD(lr)
        self.initializer = XavierInitializer()
        self.conv1 = Conv1d(in_channels=1, out_channels=4, kernel_size=3,
                            initializer=self.initializer, optimizer=self.optimizer)
        self.activation1 = ReLU()
        # flatten conv output: 4 channels * (28 - 3 + 1) = 4 * 26 = 104
        self.fc1 = FC(4*26, 128, initializer=self.initializer, optimizer=self.optimizer)  # flatten conv output
        self.fc2 = FC(128, 10, initializer=self.initializer, optimizer=self.optimizer)
        self.loss_fn = Softmax()

    def forward(self, X, Y):
        out = self.conv1.forward(X)                          # shape (4,26)
        out = self.activation1.forward(out)
        out_flat = out.flatten().reshape(1, -1)              # shape (1,104)
        out = self.fc1.forward(out_flat)                     # shape (1,128)
        out = self.fc2.forward(out)                          # shape (1,10)
        probs = self.loss_fn.forward(out)
        loss = -np.sum(Y * np.log(probs + 1e-9))
        return loss, out

    def backward(self, Y):
        dA = self.loss_fn.backward(Y)                        # (1,10)
        dA = self.fc2.backward(dA)                           # (1,128)
        dA = self.fc1.backward(dA)                           # (1,104)
        dA = dA.reshape(self.conv1.out_channels, self.conv1.N_out)  # (4,26)
        dA = self.activation1.backward(dA)
        self.conv1.backward(dA)

    def predict(self, X):
        out = self.conv1.forward(X)
        out = self.activation1.forward(out)
        out = self.fc1.forward(out.flatten().reshape(1,-1))
        out = self.fc2.forward(out)
        # return scalar label
        return int(np.argmax(out, axis=1)[0])

    def fit(self, X_train, Y_train, epochs=3):
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for i in range(len(X_train)):
                X = X_train[i]   # shape (in_channels, length)
                Y = Y_train[i].reshape(1,-1)
                loss, out = self.forward(X, Y)
                total_loss += loss
                self.backward(Y)
                # out is logits (1,10) -> take argmax along axis=1
                if np.argmax(out, axis=1)[0] == np.argmax(Y):
                    correct += 1
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train):.4f}, Accuracy: {correct/len(X_train):.4f}")

# Problem 3: Verification (MNIST)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 1, 28).astype(np.float32)/255
x_test = x_test.reshape(-1, 1, 28).astype(np.float32)/255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train small subset for demonstration
model = Scratch1dCNNClassifier(lr=0.01)
model.fit(x_train[:500], y_train[:500], epochs=3)

# Test on small subset
correct = 0
for i in range(len(x_test[:100])):
    pred = model.predict(x_test[i])
    if pred == np.argmax(y_test[i]):
        correct += 1
print(f"Test Accuracy: {correct/100:.4f}")

# %%
