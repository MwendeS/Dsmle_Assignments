# %% 
# 1. IMPORT LIBRARIES
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb, reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Embedding, Dense, Dropout, ConvLSTM2D, Flatten
from tensorflow.keras.preprocessing import sequence
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 2. DATA PREPARATION FOR TEXT SEQUENCES (IMDB)
max_features = 5000  # number of words to consider as features
maxlen = 100  # cut texts after this number of words
batch_size = 32

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("Padding sequences...")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# 3. FUNCTION TO BUILD AND TRAIN RNN MODELS
def train_rnn_model(rnn_type="SimpleRNN", units=32, epochs=3):
    """
    rnn_type: "SimpleRNN", "GRU", "LSTM"
    """
    model = Sequential()
    model.add(Embedding(max_features, 32, input_length=maxlen))
    
    if rnn_type == "SimpleRNN":
        model.add(SimpleRNN(units))
    elif rnn_type == "GRU":
        model.add(GRU(units))
    elif rnn_type == "LSTM":
        model.add(LSTM(units))
    else:
        raise ValueError("Unknown RNN type")
    
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    print(f"\nTraining {rnn_type} model...")
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=2)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"{rnn_type} Test Accuracy: {score[1]:.4f}")
    return score[1]

# 4. TRAIN AND COMPARE RNN MODELS
rnn_types = ["SimpleRNN", "GRU", "LSTM"]
results = {}

for rnn in rnn_types:
    acc = train_rnn_model(rnn_type=rnn, units=32, epochs=3)
    results[rnn] = acc

print("\nComparison of RNN Models on IMDB Dataset:")
for rnn, acc in results.items():
    print(f"{rnn}: {acc:.4f}")

# 5. CONVLSTM2D EXAMPLE
# ConvLSTM2D expects sequences of images: (samples, time_steps, rows, cols, channels)
print("\nConvLSTM2D example with dummy data...")
time_steps, rows, cols, channels = 5, 8, 8, 1
samples = 100

X = np.random.rand(samples, time_steps, rows, cols, channels)
y = np.random.randint(0, 2, samples)

conv_model = Sequential()
conv_model.add(ConvLSTM2D(filters=16, kernel_size=(3,3), input_shape=(time_steps, rows, cols, channels)))
conv_model.add(Flatten())
conv_model.add(Dense(1, activation='sigmoid'))

conv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
conv_model.fit(X, y, epochs=3, batch_size=16, verbose=2)

score = conv_model.evaluate(X, y, verbose=0)
print(f"ConvLSTM2D Dummy Data Accuracy: {score[1]:.4f}")

# 6. EXPLANATION OF OTHER CLASSES
"""
Other Keras Recurrent-Related Classes:

1. RNN: Base class for all RNN layers.
2. SimpleRNNCell, GRUCell, LSTMCell: Individual cells used to build custom RNNs.
3. StackedRNNCells: Stack multiple RNN cells manually.
4. CuDNNGRU, CuDNNLSTM: GPU-optimized versions of GRU and LSTM for faster training (requires CUDA).
"""

print("\nOther classes explained: see the docstring above.")
