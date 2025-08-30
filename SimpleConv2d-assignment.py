# %%
# # Problem 1: Creating a 2D Convolution Layer (Conv2d)
import numpy as np

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weights = np.random.randn(out_channels, in_channels, *self.kernel_size) * 0.1
        self.bias = np.zeros(out_channels)
        
    def forward(self, x):
        self.x = x
        n_samples, c, h, w = x.shape
        fh, fw = self.kernel_size
        
        # Output dimensions
        out_h = (h + 2*self.padding - fh) // self.stride + 1
        out_w = (w + 2*self.padding - fw) // self.stride + 1
        out = np.zeros((n_samples, self.out_channels, out_h, out_w))
        
        # Padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        else:
            x_padded = x
        
        # Convolution
        for n in range(n_samples):
            for m in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        region = x_padded[n, :, i*self.stride:i*self.stride+fh, j*self.stride:j*self.stride+fw]
                        out[n, m, i, j] = np.sum(region * self.weights[m]) + self.bias[m]
        return out


# Problem 2: Experiment with 2D convolution on small arrays
x = np.array([[[[ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16]]]])   # shape (1,1,4,4)

w = np.array([[[[ 0.,  0.,  0.],
                [ 0.,  1.,  0.],
                [ 0., -1.,  0.]]],

              [[[ 0.,  0.,  0.],
                [ 0., -1.,  1.],
                [ 0.,  0.,  0.]]]])    # shape (2,1,3,3)

conv_test = Conv2d(in_channels=1, out_channels=2, kernel_size=3)
conv_test.weights = w
conv_test.bias = np.zeros(2)
out_test = conv_test.forward(x)
print("Problem 2 Forward Output:\n", out_test)


# Problem 3: Function to calculate output size
def conv2d_output_size(Nh_in, Nw_in, Fh, Fw, Sh, Sw, Ph, Pw):
    Nh_out = (Nh_in + 2*Ph - Fh)//Sh + 1
    Nw_out = (Nw_in + 2*Pw - Fw)//Sw + 1
    return Nh_out, Nw_out

print("Problem 3 Example:", conv2d_output_size(28, 28, 3, 3, 1, 1, 0, 0))


# Problem 4: Max Pooling Layer
class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size)//self.stride + 1
        out_w = (w - self.kernel_size)//self.stride + 1
        out = np.zeros((n, c, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                out[:, :, i, j] = np.max(region, axis=(2,3))
        return out


# Problem 5: Average Pooling Layer
class AveragePool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size)//self.stride + 1
        out_w = (w - self.kernel_size)//self.stride + 1
        out = np.zeros((n, c, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                out[:, :, i, j] = np.mean(region, axis=(2,3))
        return out


# Problem 6: Flatten Layer
class Flatten:
    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)


# Problem 7: Simple CNN Model for MNIST
class Dense:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias

class Scratch2dCNNClassifier:
    def __init__(self):
        self.conv1 = Conv2d(1, 8, 3)        # Conv layer
        self.pool = MaxPool2D(2,2)          # Max pooling
        self.flatten = Flatten()            # Flatten
        self.fc1 = Dense(8*13*13, 10)       # Fully connected

    def forward(self, x):
        x = self.conv1.forward(x)
        x = np.maximum(x, 0)  # ReLU
        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        return x


# Load MNIST dataset
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,1,28,28)/255.0
x_test = x_test.reshape(-1,1,28,28)/255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Simple Training Loop (forward only, no backprop yet)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-7)) / y_pred.shape[0]

model = Scratch2dCNNClassifier()
y_pred = model.forward(x_train[:100])  # only 100 samples for quick test
y_pred_soft = softmax(y_pred)
loss = cross_entropy(y_pred_soft, y_train[:100])
print("Problem 7: Training Forward Pass - Loss:", loss)

# %%
