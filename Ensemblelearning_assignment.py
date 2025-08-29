# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ScratchKMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    # Step 1: Initialize centers
    def _init_centers(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[random_idx]

    # Step 2: Assign points to nearest center
    def _assign_clusters(self, X, centers):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        return np.argmin(distances, axis=1)

    # Step 3: Update centers based on assigned points
    def _update_centers(self, X, labels, old_centers):
        new_centers = []
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                # If no points in a cluster, keep old center
                new_centers.append(old_centers[k])
        return np.array(new_centers)

    # Step 4: Run the algorithm
    def fit(self, X):
        centers = self._init_centers(X)

        for i in range(self.max_iter):
            labels = self._assign_clusters(X, centers)
            new_centers = self._update_centers(X, labels, centers)

            # Check movement of centers
            shift = np.sum(np.linalg.norm(centers - new_centers, axis=1))
            centers = new_centers

            if shift == 0:  # convergence reached
                break

        self.cluster_centers_ = centers
        self.labels_ = labels

    # Step 5: Predict for new data
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)


# Example Run with Dataset
# Load Wholesale Customers dataset
data_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\Housing information data\train.csv"
# Use only selected features
X = data[['Fresh', 'Milk', 'Grocery', 'Frozen','Detergents_Paper', 'Delicassen']].values
# Run KMeans
kmeans = ScratchKMeans(n_clusters=3, max_iter=100, random_state=42)
kmeans.fit(X)
print("Final cluster centers:\n", kmeans.cluster_centers_)
print("Labels for first 10 points:", kmeans.labels_[:10])

# Simple 2D visualization using first two features
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, label="Centers")
plt.xlabel("Fresh")
plt.ylabel("Milk")
plt.legend()
plt.title("KMeans Clustering from Scratch")
plt.show()
# %%
