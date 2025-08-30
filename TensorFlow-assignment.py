# %%
# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, fetch_openml

# 2. BINARY CLASSIFICATION: IRIS (2 CLASSES)
print("Binary classification: Iris (versicolor vs virginica)")

# Load Iris dataset from sklearn
iris = load_iris(as_frame=True)
df = iris.frame
df = df[(df["target"] == 1) | (df["target"] == 2)]  # 1=versicolor, 2=virginica
X = df[iris.feature_names].values
y = np.where(df["target"] == 1, 0, 1)[:, np.newaxis]

# Split dataset: train / val / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Hyperparameters
learning_rate = 0.001
num_epochs = 100
batch_size = 10
n_input = X_train.shape[1]
n_hidden1 = 50
n_hidden2 = 100
n_classes = 1  # binary

# Placeholders
tf.compat.v1.disable_eager_execution()
X_ph = tf.compat.v1.placeholder(tf.float32, [None, n_input])
Y_ph = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

# Neural network definition
def simple_net(x, n_input, n_hidden1, n_hidden2, n_classes):
    weights = {
        'w1': tf.Variable(tf.random.normal([n_input, n_hidden1])),
        'w2': tf.Variable(tf.random.normal([n_hidden1, n_hidden2])),
        'w3': tf.Variable(tf.random.normal([n_hidden2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden1])),
        'b2': tf.Variable(tf.random.normal([n_hidden2])),
        'b3': tf.Variable(tf.random.normal([n_classes]))
    }

    l1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, weights['w2']), biases['b2']))
    output = tf.add(tf.matmul(l2, weights['w3']), biases['b3'])
    return output

logits = simple_net(X_ph, n_input, n_hidden1, n_hidden2, n_classes)

# Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_ph, logits=logits))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# Accuracy
pred = tf.sigmoid(logits)
correct_pred = tf.equal(tf.round(pred), Y_ph)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize variables
init = tf.compat.v1.global_variables_initializer()

# Train model
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        # mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={X_ph: batch_x, Y_ph: batch_y})

        val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={X_ph: X_val, Y_ph: y_val})
        if epoch % 10 == 0:
            print("Epoch {}: val_loss {:.4f}, val_acc {:.3f}".format(epoch, val_loss, val_acc))
    
    test_acc = sess.run(accuracy, feed_dict={X_ph: X_test, Y_ph: y_test})
    print("Test Accuracy: {:.3f}".format(test_acc))


# 3. MULTI-CLASS CLASSIFICATION: IRIS (3 CLASSES)
print("\nMulti-class classification: Iris (all 3 species)")

df = iris.frame
X = df[iris.feature_names].values
y = df["target"].values
y = np.eye(3)[y]  # one-hot encoding

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

n_classes = 3
Y_ph = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
logits = simple_net(X_ph, n_input, n_hidden1, n_hidden2, n_classes)

# For multi-class classification, use softmax cross-entropy
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_ph, logits=logits))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

pred = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_ph, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        for i in range(0, X_train.shape[0], batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={X_ph: batch_x, Y_ph: batch_y})

        val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={X_ph: X_val, Y_ph: y_val})
        if epoch % 10 == 0:
            print("Epoch {}: val_loss {:.4f}, val_acc {:.3f}".format(epoch, val_loss, val_acc))
    
    test_acc = sess.run(accuracy, feed_dict={X_ph: X_test, Y_ph: y_test})
    print("Test Accuracy: {:.3f}".format(test_acc))

# %%
