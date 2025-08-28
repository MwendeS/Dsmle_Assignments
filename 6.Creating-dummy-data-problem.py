
# %%
# Assignment: NumPy & Visualization Practice
# Purpose:
# 1. Get familiar with NumPy for random data creation
# 2. Practice graph visualization using matplotlib
import numpy as np
import matplotlib.pyplot as plt
# [Problem 1] Create random numbers
np.random.seed(0)  # fix seed for reproducibility

mean1 = [-3, 0]
cov = [[1.0, 0.8], 
       [0.8, 1.0]]  # covariance matrix

data1 = np.random.multivariate_normal(mean1, cov, 500)
print("Problem 1: data1 shape =", data1.shape)  # (500, 2)
# [Problem 2] Scatter plot visualization
plt.figure(figsize=(6, 6))
plt.scatter(data1[:, 0], data1[:, 1], alpha=0.6)
plt.title("Problem 2: Scatter Plot of Data1")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
# [Problem 3] Histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data1[:, 0], bins=30, alpha=0.7, color="blue")
plt.title("Histogram of X (dimension 0)")
plt.xlim(-7, 5)
plt.subplot(1, 2, 2)
plt.hist(data1[:, 1], bins=30, alpha=0.7, color="green")
plt.title("Histogram of Y (dimension 1)")
plt.xlim(-7, 5)
plt.tight_layout()
plt.show()
# [Problem 4] Add new data
mean2 = [0, -3]
data2 = np.random.multivariate_normal(mean2, cov, 500)
print("Problem 4: data2 shape =", data2.shape)  # (500, 2)

plt.figure(figsize=(6, 6))
plt.scatter(data1[:, 0], data1[:, 1], alpha=0.6, label="Class 0")
plt.scatter(data2[:, 0], data2[:, 1], alpha=0.6, label="Class 1")
plt.title("Problem 4: Scatter Plot of Data1 & Data2")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()
# [Problem 5] Data Combination
combined_data = np.vstack((data1, data2))
print("Problem 5: combined_data shape =", combined_data.shape)  # (1000, 2)

# [Problem 6] Labeling
labels1 = np.zeros((500, 1))  # label 0 for data1
labels2 = np.ones((500, 1))   # label 1 for data2

labels = np.vstack((labels1, labels2))  # (1000, 1)

final_dataset = np.hstack((combined_data, labels))
print("Problem 6: final_dataset shape =", final_dataset.shape)  # (1000, 3)
# Preview of final dataset
print("First 5 rows of dataset:\n", final_dataset[:5])
# %%
