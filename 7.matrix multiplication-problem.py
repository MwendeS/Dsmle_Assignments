# %%# Assignment: Matrix Multiplication with NumPy
# Goal:
# 1. Understand basic matrix operations using NumPy
# 2. Implement matrix multiplication from scratch
import numpy as np
# [Step 1] Define matrices A and B
a_ndarray = np.array([[-1, 2, 3],
                      [4, -5, 6],
                      [7, 8, -9]])

b_ndarray = np.array([[0, 2, 1],
                      [0, 2, -8],
                      [2, 9, -1]])

print("Matrix A:\n", a_ndarray)
print("Matrix B:\n", b_ndarray)
# [Step 2] Matrix multiplication using NumPy
result_numpy = np.matmul(a_ndarray, b_ndarray)
# Alternative: result_numpy = a_ndarray @ b_ndarray
print("\n[NumPy Result]\n", result_numpy)
# [Step 3] Scratch implementation of matrix multiplication
def matmul_scratch(a, b):
    result = np.zeros((a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):      # loop over rows of A
        for j in range(b.shape[1]):  # loop over columns of B
            for k in range(a.shape[1]):  # loop over columns of A / rows of B
                result[i, j] += a[i, k] * b[k, j]
    return result

scratch_result = matmul_scratch(a_ndarray, b_ndarray)
print("\n[Scratch Result]\n", scratch_result)
# Check if both results are equal
print("\nAre NumPy result and Scratch result equal? ->", np.allclose(result_numpy, scratch_result))
# [Step 4] What happens if no calculation is defined?
d_ndarray = np.array([[-1, 2, 3],
                      [4, -5, 6]])  # shape (2, 3)

e_ndarray = np.array([[-9, 8, 7, 6],
                      [-5, 4, 3, 2]])  # shape (2, 4)

print("\nMatrix D shape:", d_ndarray.shape)
print("Matrix E shape:", e_ndarray.shape)

def is_multiplicable(a, b):
    if a.shape[1] != b.shape[0]:
        print("Cannot multiply: column count of A != row count of B")
        return False
    return True

if is_multiplicable(d_ndarray, e_ndarray):
    print(np.matmul(d_ndarray, e_ndarray))
# [Step 5] Multiplication by using transpose
e_transposed = e_ndarray.T  # transpose
print("\nE Transposed shape:", e_transposed.shape)

if is_multiplicable(d_ndarray, e_transposed):
    transposed_result = np.matmul(d_ndarray, e_transposed)
    print("\n[Result using transpose]\n", transposed_result)
# [Summary]
print("\nSummary:")
print("1. Matrix multiplication is a key operation in ML & data analysis.")
print("2. NumPy makes it easy and efficient.")
print("3. Scratch implementation deepens understanding.")
print("4. Multiplication is only possible if cols(A) == rows(B).")
print("5. Transpose can sometimes make multiplication possible, but meaning changes.")

# %%
