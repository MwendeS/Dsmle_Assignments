# %% CLUSTERING
# Tasks:
# Implement K-means algorithm.
# Test it on artificial data (like blobs).
# Then apply it on real data (Wholesale customers dataset).
# Try elbow method & silhouette score to decide cluster number.
# Compare clusters with real groups (Region/Channel).
# Write useful insights for wholesalers.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd

class ScratchKMeans:
    def __init__(self, n_clusters=3, n_init=10, max_iter=100, tol=1e-4, verbose=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None  # SSE

    def _init_centers(self, X):
        """Randomly pick initial centers from data"""
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _compute_sse(self, X, labels, centers):
        """Calculate SSE"""
        sse = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            sse += np.sum((cluster_points - centers[k])**2)
        return sse

    def _assign_clusters(self, X, centers):
        """Assign each point to nearest center"""
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centers(self, X, labels, centers):
        """Move centers to mean of assigned points"""
        new_centers = []
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                new_centers.append(centers[k]) # keep the old one
        return np.array(new_centers)

    def fit(self, X):
        best_sse = float("inf")
        best_centers, best_labels = None, None

        for _ in range(self.n_init):
            centers = self._init_centers(X)
            for i in range(self.max_iter):
                labels = self._assign_clusters(X, centers)
                new_centers = self._update_centers(X, labels, centers) # fixed
                shift = np.sum(np.linalg.norm(centers - new_centers, axis=1))

                centers = new_centers
                if self.verbose:
                    print(f"Iteration {i}, shift={shift:.4f}")

                if shift < self.tol:
                    break

            sse = self._compute_sse(X, labels, centers)
            if sse < best_sse:
                best_sse = sse
                best_centers = centers
                best_labels = labels

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_sse

    def predict(self, X):
        """Assign new data to nearest cluster"""
        return self._assign_clusters(X, self.cluster_centers_)


# ------------------------------
# TEST ON ARTIFICIAL DATA
# ------------------------------
X, _ = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=0.5, random_state=0)

model = ScratchKMeans(n_clusters=4, n_init=5, max_iter=100, tol=1e-4, verbose=True)
model.fit(X)

plt.scatter(X[:,0], X[:,1], c=model.labels_, cmap='viridis')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c="red", marker="x", s=200)
plt.title("Scratch K-means on Artificial Data")
plt.show()


# ------------------------------
# ELBOW METHOD
# ------------------------------
sse_list = []
for k in range(1, 10):
    km = ScratchKMeans(n_clusters=k, n_init=5, max_iter=100, tol=1e-4)
    km.fit(X)
    sse_list.append(km.inertia_)

plt.plot(range(1,10), sse_list, marker='o')
plt.xlabel("Number of clusters k")
plt.ylabel("SSE (inertia)")
plt.title("Elbow Method")
plt.show()


# ------------------------------
# WHOLESALE DATA
# ------------------------------
# data = pd.read_csv("Wholesale customers data.csv")
data = pd.read_csv(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\Wholesale customers data.csv")
data = data.drop(["Channel", "Region"], axis=1)

# PCA for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Fit model
km = ScratchKMeans(n_clusters=3, n_init=10, max_iter=100, tol=1e-4)
km.fit(data.values)

plt.scatter(data_pca[:,0], data_pca[:,1], c=km.labels_, cmap="plasma")
plt.title("Wholesale Customers Clusters (PCA reduced)")
plt.show()

# Silhouette Score
score = silhouette_score(data, km.labels_)
print("Silhouette Score:", score)
# %%
