# %%
# Assignment: Gradients and Functions with NumPy
# Goal:
# 1. Get familiar with mathematical operations using NumPy
# 2. Develop gradient-finding skills
import numpy as np
import matplotlib.pyplot as plt
# [Problem 1] Create x and y for linear function y = (1/2)x + 1
x = np.arange(-50, 50.1, 0.1)  # from -50 to 50 step 0.1
y = 0.5 * x + 1
# [Problem 2] Array concatenation into shape (n, 2)
array_xy = np.column_stack((x, y))
print("Problem 2: array_xy shape =", array_xy.shape)
# [Problem 3] Gradient calculation (finite differences)
gradient = np.diff(y) / np.diff(x)  # slope between adjacent points
print("Problem 3: gradient shape =", gradient.shape)
# [Problem 4] Graph of function and gradient
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y, label="y = 0.5x + 1")
plt.title("Linear Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x[:-1], gradient, label="Gradient", color="red")
plt.title("Gradient of Linear Function")
plt.xlabel("x")
plt.ylabel("dy/dx (approx.)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# [Problem 5] General compute_gradient function
def compute_gradient(function, x_range=(-50, 50.1, 0.1)):
    """
    Compute function values and gradient using finite differences.
    Parameters:
    ----------------
    function : callable
        Function that maps ndarray x -> ndarray y
    x_range : tuple
        (start, stop, step) passed to np.arange()

    Returns
    array_xy : ndarray, shape (n, 2)
        x and y combined
    gradient : ndarray, shape (n-1,)
        gradient values
    """
    x = np.arange(*x_range)
    y = function(x)
    array_xy = np.column_stack((x, y))
    gradient = np.diff(y) / np.diff(x)
    return array_xy, gradient
# Define functions
def function1(x):
    return x ** 2
def function2(x):
    return 2 * x ** 2 + 2 * x
def function3(x):
    return np.sin(x / 2)
# Test each function
fns = [(function1, (-50, 50.1, 0.1), "y = x^2"),
       (function2, (-50, 50.1, 0.1), "y = 2x^2 + 2x"),
       (function3, (0, 50.1, 0.1), "y = sin(x/2)")]

for fn, xr, title in fns:
    arr, grad = compute_gradient(fn, xr)
    x_vals, y_vals = arr[:, 0], arr[:, 1]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, label=title)
    plt.title(f"Function: {title}")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x_vals[:-1], grad, label="Gradient", color="red")
    plt.title(f"Gradient of {title}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# [Problem 6] Finding minimum values
for fn, xr, title in fns:
    arr, grad = compute_gradient(fn, xr)
    y_vals = arr[:, 1]

    min_val = y_vals.min()
    min_idx = y_vals.argmin()
    x_min = arr[min_idx, 0]

    print(f"\nFunction: {title}")
    print(f"Minimum y = {min_val:.4f} at x = {x_min}")
    if 0 < min_idx < len(grad):
        print("Gradient before min:", grad[min_idx - 1])
        print("Gradient after  min:", grad[min_idx])
    else:
        print("Gradient check skipped (min at edge).")
# [Summary]
print("\nSummary:")
print("1. NumPy allows efficient calculation of functions & gradients.")
print("2. Gradient = rate of change, helps find minima/maxima.")
print("3. Visualizing function & gradient gives intuition.")
print("4. Found minima using min() and argmin() instead of gradient descent.")

# %%
