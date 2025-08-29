
# %%
import numpy as np
import matplotlib.pyplot as plt
# Mean Squared Error Function
def MSE(y_pred, y):
    return np.mean((y_pred - y) ** 2)
# Scratch Linear Regression Class
class ScratchLinearRegression:
    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.coef_ = None
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    # Hypothesis function
    def _linear_hypothesis(self, X):
        return X.dot(self.coef_)

    # Gradient descent update
    def _gradient_descent(self, X, error):
        gradient = X.T.dot(error) / X.shape[0]
        self.coef_ -= self.lr * gradient
    # Train the model
    def fit(self, X, y, X_val=None, y_val=None):
        if not self.no_bias:
            X = np.c_[np.ones(X.shape[0]), X]
            if X_val is not None:
                X_val = np.c_[np.ones(X_val.shape[0]), X_val]

        self.coef_ = np.zeros(X.shape[1])

        for i in range(self.iter):
            y_pred = self._linear_hypothesis(X)
            error = y_pred - y
            self.loss[i] = MSE(y_pred, y) / 2  # objective function

            self._gradient_descent(X, error)

            if X_val is not None and y_val is not None:
                y_val_pred = self._linear_hypothesis(X_val)
                self.val_loss[i] = MSE(y_val_pred, y_val) / 2

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {self.loss[i]}")

    # Predict new data
    def predict(self, X):
        if not self.no_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        return self._linear_hypothesis(X)

# Example: Train and plot learning curve
if __name__ == "__main__":
    # Example data (replace with your dataset)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=True)
    model.fit(X, y)
    # Predictions
    y_pred = model.predict(X)
    print("Predictions:", y_pred)
    # Learning curve
    plt.plot(model.loss, label="Train Loss")
    if model.val_loss.any():
        plt.plot(model.val_loss, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()
# %%
