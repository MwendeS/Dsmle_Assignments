# %%
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Problem 1: Gini impurity
def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

# Problem 2: Information gain
def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)
    if n_left == 0 or n_right == 0:
        return 0
    gain = gini_impurity(y_parent) - (n_left/n) * gini_impurity(y_left) - (n_right/n) * gini_impurity(y_right)
    return gain

# Problem 3 & 4: Scratch Decision Tree (Depth 1)
class ScratchDecisionTreeClassifierDepth1():
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.feature = None
        self.threshold = None
        self.l_label = None
        self.r_label = None

    def fit(self, X, y):
        best_gain = 0
        n_samples, n_features = X.shape
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for t in thresholds:
                left_idx = X[:, feature_index] < t
                right_idx = ~left_idx
                y_left, y_right = y[left_idx], y[right_idx]
                gain = information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    self.feature = feature_index
                    self.threshold = t
                    self.l_label = collections.Counter(y_left).most_common(1)[0][0]
                    self.r_label = collections.Counter(y_right).most_common(1)[0][0]
                    if self.verbose:
                        print(f"Feature {feature_index}, Threshold {t}, Gain {gain:.3f}")

    def predict(self, X):
        return np.where(X[:, self.feature] < self.threshold, self.l_label, self.r_label)

# Problem 5: Verification
# Simple dataset
data, labels = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Train Scratch model
scratch_clf = ScratchDecisionTreeClassifierDepth1(verbose=True)
scratch_clf.fit(X_train, y_train)
y_pred_train = scratch_clf.predict(X_train)

print("Scratch Model Accuracy:", accuracy_score(y_train, y_pred_train))
print("Scratch Model Precision:", precision_score(y_train, y_pred_train))
print("Scratch Model Recall:", recall_score(y_train, y_pred_train))

# Compare with sklearn
sk_clf = DecisionTreeClassifier(max_depth=1, random_state=42)
sk_clf.fit(X_train, y_train)
sk_pred_train = sk_clf.predict(X_train)

print("Sklearn Accuracy:", accuracy_score(y_train, sk_pred_train))
print("Sklearn Precision:", precision_score(y_train, sk_pred_train))
print("Sklearn Recall:", recall_score(y_train, sk_pred_train))

# Problem 6: Decision region visualization
def plot_decision_regions(X, y, classifier, resolution=0.02):
    from matplotlib.colors import ListedColormap
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=40, cmap=cmap)
    plt.show()

plot_decision_regions(X_train, y_train, scratch_clf)
# %%
