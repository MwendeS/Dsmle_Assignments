# %% 
# 1. IMPORT LIBRARIES
import numpy as np


# 2. DEFINE SIMPLE RNN CLASS
class ScratchSimpleRNNClassifier:
    def __init__(self, n_features, n_nodes, n_outputs, learning_rate=0.01):
        """
        n_features: number of input features
        n_nodes: number of hidden nodes in RNN
        n_outputs: number of output classes
        """
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.lr = learning_rate

        # Initialize weights and bias
        self.Wx = np.random.randn(n_features, n_nodes) * 0.01  # Input weights
        self.Wh = np.random.randn(n_nodes, n_nodes) * 0.01     # Hidden state weights
        self.Bh = np.zeros((1, n_nodes))                        # Bias
        self.Wy = np.random.randn(n_nodes, n_outputs) * 0.01   # Output weights
        self.By = np.zeros((1, n_outputs))                     # Output bias

    # Activation function
    def tanh(self, x):
        return np.tanh(x)
    
    # Derivative of tanh
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    # Softmax function for output
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Forward propagation
    def forward(self, X):
        """
        X shape: (batch_size, n_sequences, n_features)
        """
        batch_size, n_sequences, _ = X.shape
        h = np.zeros((batch_size, self.n_nodes))  # initial hidden state

        self.h_history = []  # store hidden states for backprop
        for t in range(n_sequences):
            x_t = X[:, t, :]  # input at time t
            a_t = np.dot(x_t, self.Wx) + np.dot(h, self.Wh) + self.Bh
            h = self.tanh(a_t)
            self.h_history.append(h)
        # Output from last time step
        y = np.dot(h, self.Wy) + self.By
        y = self.softmax(y)
        return y

    # Simple cross-entropy loss
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    # Backpropagation (simplified version)
    def backward(self, X, y_true, y_pred):
        batch_size, n_sequences, _ = X.shape
        
        # Gradients initialization
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dBh = np.zeros_like(self.Bh)
        dWy = np.zeros_like(self.Wy)
        dBy = np.zeros_like(self.By)
        
        # Output gradient
        dy = y_pred.copy()
        dy[range(batch_size), y_true] -= 1
        dy /= batch_size
        
        # Gradient for Wy and By
        h_last = self.h_history[-1]
        dWy = np.dot(h_last.T, dy)
        dBy = np.sum(dy, axis=0, keepdims=True)
        
        # Gradient for hidden layers
        dh_next = np.dot(dy, self.Wy.T)
        for t in reversed(range(n_sequences)):
            h = self.h_history[t]
            dh = dh_next
            da = dh * self.tanh_derivative(h)
            
            x_t = X[:, t, :]
            h_prev = self.h_history[t-1] if t > 0 else np.zeros_like(h)
            
            dWx += np.dot(x_t.T, da)
            dWh += np.dot(h_prev.T, da)
            dBh += np.sum(da, axis=0, keepdims=True)
            
            dh_next = np.dot(da, self.Wh.T)
        
        # Update weights
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        self.Bh -= self.lr * dBh
        self.Wy -= self.lr * dWy
        self.By -= self.lr * dBy

    # Training function
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y, y_pred)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


# 3. TEST THE RNN WITH SMALL ARRAY
x = np.array([[[1, 2], [2, 3], [3, 4]]])/100  # (batch_size=1, n_sequences=3, n_features=2)
y = np.array([0])  # target class

# Create RNN instance
rnn = ScratchSimpleRNNClassifier(n_features=2, n_nodes=4, n_outputs=2, learning_rate=0.1)

# Forward propagation
y_pred = rnn.forward(x)
print("Predicted output (softmax probabilities):", y_pred)

# Optional: Train for a few epochs
rnn.fit(x, y, epochs=50)

# Forward again to see improvement
y_pred = rnn.forward(x)
print("Predicted output after training:", y_pred)

# %%
