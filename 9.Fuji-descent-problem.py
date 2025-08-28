# %%
# Assignment: Gradient Descent - Mount Fuji Problem
# Goal:
# 1. Get familiar with NumPy
# 2. Learn how gradient descent works through Mt. Fuji analogy
import numpy as np
import matplotlib.pyplot as plt
# Load Data
# csv_path = "mtfuji_data.csv"  # place the csv file in the same directory
csv_path = r"c:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\mtfuji_data.csv"
np.set_printoptions(suppress=True)
fuji = np.loadtxt(csv_path, delimiter=",", skiprows=1)
# fuji[:, 0] -> point number
# fuji[:, 1] -> latitude
# fuji[:, 2] -> longitude
# fuji[:, 3] -> elevation
# fuji[:, 4] -> distance from start (m)
print("Fuji data shape:", fuji.shape)
# [Problem 1] Data visualization
plt.figure(figsize=(10, 5))
plt.plot(fuji[:, 0], fuji[:, 3], label="Mt. Fuji Cross-section")
plt.xlabel("Point Number")
plt.ylabel("Elevation (m)")
plt.title("Cross-section of Mt. Fuji")
plt.grid(True)
plt.legend()
plt.show()
# [Problem 2] Gradient at a point
def compute_gradient(point, fuji):
    """
    Compute gradient at a given point index.
    dy/dx = (y[i] - y[i-1]) / (x[i] - x[i-1])
    """
    if point <= 0 or point >= len(fuji):
        return None
    x = fuji[:, 0]
    y = fuji[:, 3]
    grad = (y[point] - y[point - 1]) / (x[point] - x[point - 1])
    return grad
# [Problem 3] Next point calculation
def next_point(point, alpha=0.2, fuji=fuji):
    grad = compute_gradient(point, fuji)
    if grad is None:
        return None
    dest = point - alpha * grad
    dest = int(round(dest))
    if dest < 0 or dest >= len(fuji):
        return None
    return dest
# [Problem 4] Go down the mountain
def descend(initial_point=136, alpha=0.2, fuji=fuji):
    path = [initial_point]
    current = initial_point
    while True:
        nxt = next_point(current, alpha, fuji)
        if nxt is None or nxt == current:
            break
        path.append(nxt)
        current = nxt
    return path
# Test: start from point 136
path_136 = descend(136)
print("Path from 136:", path_136[:10], "... total steps:", len(path_136))
# [Problem 5] Visualization of descent process
def visualize_descent(path, fuji, title="Descent process"):
    elevations = fuji[:, 3]
    # Plot cross-section with path points
    plt.figure(figsize=(10, 5))
    plt.plot(fuji[:, 0], elevations, label="Mt. Fuji")
    plt.scatter(path, elevations[path], c="red", label="Descent path")
    plt.xlabel("Point Number")
    plt.ylabel("Elevation (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot gradient values
    grads = [compute_gradient(p, fuji) for p in path if compute_gradient(p, fuji) is not None]
    plt.figure(figsize=(10, 5))
    plt.plot(grads, marker="o", label="Gradient at each step")
    plt.xlabel("Step")
    plt.ylabel("Gradient")
    plt.title(f"Gradient evolution: {title}")
    plt.legend()
    plt.grid(True)
    plt.show()
# Visualize descent from point 136
visualize_descent(path_136, fuji, "Descent from point 136")
# [Problem 6] Changing initial value
path_142 = descend(142)
visualize_descent(path_142, fuji, "Descent from point 142")
# [Problem 7] Compare descents from multiple initial values
init_points = [50, 100, 136, 142, 200]
plt.figure(figsize=(10, 5))
plt.plot(fuji[:, 0], fuji[:, 3], label="Mt. Fuji")
for start in init_points:
    path = descend(start)
    plt.scatter(path, fuji[path, 3], label=f"Start={start}")
plt.xlabel("Point Number")
plt.ylabel("Elevation (m)")
plt.title("Descent paths from different initial points")
plt.legend()
plt.grid(True)
plt.show()
# [Problem 8] Changing hyperparameter alpha
alphas = [0.05, 0.2, 0.5, 1.0]
plt.figure(figsize=(10, 5))
plt.plot(fuji[:, 0], fuji[:, 3], label="Mt. Fuji")

for a in alphas:
    path = descend(136, alpha=a)
    plt.scatter(path, fuji[path, 3], label=f"alpha={a}")

plt.xlabel("Point Number")
plt.ylabel("Elevation (m)")
plt.title("Effect of learning rate (alpha) on descent")
plt.legend()
plt.grid(True)
plt.show()
# [Summary]
print("\nSummary:")
print("1. Mt. Fuji elevation data visualized as cross-section.")
print("2. Gradient descent simulated as descending the mountain.")
print("3. Descent depends on initial value (starting point).")
print("4. Learning rate (alpha) strongly affects descent behavior.")
print("5. This analogy helps understand gradient descent in ML.")

# %%
