# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression as SkLogReg
import pickle
import os

# --------------------------
# Logistic Regression (Scratch)
# --------------------------
class ScratchLogisticRegression():
    def __init__(self, num_iter=1000, lr=0.1, bias=True, verbose=False, reg_lambda=0.01):
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.reg_lambda = reg_lambda
        self.loss = []
        self.val_loss = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss_function(self, X, y, theta):
        m = len(y)
        h = self._sigmoid(X.dot(theta))
        epsilon = 1e-10
        loss = (-1/m) * np.sum(y*np.log(h+epsilon) + (1-y)*np.log(1-h+epsilon))
        reg = (self.reg_lambda/(2*m)) * np.sum(theta[1:]**2)  # no reg on bias
        return loss + reg

    def fit(self, X, y, X_val=None, y_val=None):
        m, n = X.shape
        if self.bias:
            X = np.c_[np.ones((m, 1)), X]
            if X_val is not None:
                X_val = np.c_[np.ones((X_val.shape[0], 1)), X_val]

        self.theta = np.zeros(X.shape[1])

        for i in range(self.iter):
            h = self._sigmoid(X.dot(self.theta))
            gradient = (1/m) * X.T.dot(h - y)
            gradient[1:] += (self.reg_lambda/m) * self.theta[1:]
            self.theta -= self.lr * gradient

            # record loss
            self.loss.append(self._loss_function(X, y, self.theta))
            if X_val is not None and y_val is not None:
                self.val_loss.append(self._loss_function(X_val, y_val, self.theta))

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Loss={self.loss[-1]}")

    def predict_proba(self, X):
        if self.bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(X.dot(self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save_weights(self, filepath):
        np.savez(filepath, theta=self.theta)

    def load_weights(self, filepath):
        data = np.load(filepath)
        self.theta = data["theta"]

# --------------------------
# Verification on Iris dataset
# --------------------------
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[iris.target != 0][:, [2, 3]]  # only petal length, petal width
    y = (iris.target[iris.target != 0] - 1)     # versicolor=0, virginica=1

    # train-test split
    np.random.seed(0)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # train scratch model
    model = ScratchLogisticRegression(num_iter=1000, lr=0.1, verbose=True)
    model.fit(X_train, y_train, X_test, y_test)

    # predictions
    y_pred = model.predict(X_test)

    print("Scratch Logistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # compare with scikit-learn
    sk_model = SkLogReg(max_iter=1000)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    print("\nScikit-learn Logistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, sk_pred))
    print("Precision:", precision_score(y_test, sk_pred))
    print("Recall:", recall_score(y_test, sk_pred))

    # learning curve
    plt.plot(model.loss, label="Train Loss")
    if model.val_loss:
        plt.plot(model.val_loss, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curve")
    plt.savefig(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\LogisticRegressionOutputs\learning_curve.png")
    plt.close()

    # decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("Decision Region (Scratch Logistic Regression)")
    plt.savefig(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\LogisticRegressionOutputs\decision_region.png")
    plt.close()

    # save weights
    model.save_weights(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\LogisticRegressionOutputs\logreg_weights.npz")
    print("\nModel weights saved.")

# %%
