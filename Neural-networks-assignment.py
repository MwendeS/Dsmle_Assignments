# %%
# Output:
# Accuracy on validation set.
# Learning curve plot.
# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Step 2: Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Normalize pixels to 0-1
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# One-hot encode labels
enc = OneHotEncoder(sparse_output=False)
y_train_one_hot = enc.fit_transform(y_train[:, np.newaxis])
y_test_one_hot = enc.transform(y_test[:, np.newaxis])

# Split training into training + validation
X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(
    X_train, y_train_one_hot, test_size=0.2, random_state=42
)

# Step 3: Define utility functions
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-7), axis=1))

# Mini-batch iterator
class GetMiniBatch:
    def __init__(self, X, y, batch_size=20):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(X.shape[0])
    def __iter__(self):
        np.random.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            idx = self.indices[start:start+self.batch_size]
            yield self.X[idx], self.y[idx]

# Step 4: Neural Network Class
class ScratchSimpleNeuralNetwork:
    def __init__(self, n_features=784, n_hidden1=400, n_hidden2=200, n_output=10, lr=0.01):
        # Initialize weights
        sigma = 0.01
        self.W1 = sigma * np.random.randn(n_features, n_hidden1)
        self.b1 = np.zeros(n_hidden1)
        self.W2 = sigma * np.random.randn(n_hidden1, n_hidden2)
        self.b2 = np.zeros(n_hidden2)
        self.W3 = sigma * np.random.randn(n_hidden2, n_output)
        self.b3 = np.zeros(n_output)
        self.lr = lr
    
    def forward(self, X):
        self.A1 = X @ self.W1 + self.b1
        self.Z1 = tanh(self.A1)
        self.A2 = self.Z1 @ self.W2 + self.b2
        self.Z2 = tanh(self.A2)
        self.A3 = self.Z2 @ self.W3 + self.b3
        self.Z3 = softmax(self.A3)
        return self.Z3
    
    def backward(self, X, y_true):
        batch_size = X.shape[0]
        dA3 = (self.Z3 - y_true) / batch_size
        dW3 = self.Z2.T @ dA3
        db3 = np.sum(dA3, axis=0)
        
        dZ2 = dA3 @ self.W3.T
        dA2 = dZ2 * tanh_derivative(self.A2)
        dW2 = self.Z1.T @ dA2
        db2 = np.sum(dA2, axis=0)
        
        dZ1 = dA2 @ self.W2.T
        dA1 = dZ1 * tanh_derivative(self.A1)
        dW1 = X.T @ dA1
        db1 = np.sum(dA1, axis=0)
        
        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def fit(self, X, y, epochs=5, batch_size=20):
        for epoch in range(epochs):
            for X_batch, y_batch in GetMiniBatch(X, y, batch_size):
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            # Compute loss
            y_pred_train = self.forward(X)
            loss = cross_entropy(y_pred_train, y)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Step 5: Train and Evaluate
nn = ScratchSimpleNeuralNetwork()
nn.fit(X_train, y_train_one_hot, epochs=5, batch_size=20)

# Accuracy on validation set
y_val_pred = nn.predict(X_val)
y_val_true = np.argmax(y_val_one_hot, axis=1)
accuracy = np.mean(y_val_pred == y_val_true)
print("Validation Accuracy:", accuracy)

# Step 6: Plot example images with predictions
num = 9
plt.figure(figsize=(6,6))
for i in range(num):
    plt.subplot(3,3,i+1)
    plt.imshow(X_val[i].reshape(28,28), cmap='gray')
    plt.title(f"Pred:{y_val_pred[i]}, True:{y_val_true[i]}")
    plt.axis('off')
plt.show()
# %%