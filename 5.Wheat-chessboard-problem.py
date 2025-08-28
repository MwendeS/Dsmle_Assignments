# %%
# Wheat and Chessboard Problem
import numpy as np
import matplotlib.pyplot as plt
def wheat_on_board(n_rows=8, n_cols=8, dtype=np.uint64):
    """
    Calculate the number of grains of wheat on an n x m chessboard.
    Each square doubles the previous square, starting from 1.

    Parameters
    ----------
    n_rows : int
        Number of rows on the board (default 8)
    n_cols : int
        Number of columns on the board (default 8)
    dtype : numpy dtype
        Data type for array, default np.uint64 to avoid overflow

    Returns
    -------
    board : ndarray
        n x m board where each element is the number of grains
    """
    n_squares = n_rows * n_cols
    indices = np.arange(n_squares, dtype=dtype)
    values = 2 ** indices
    board = values.reshape((n_rows, n_cols))
    return board

def analyze_board(n_rows=8, n_cols=8):
    """Perform analysis and visualization for the chessboard problem."""
    board = wheat_on_board(n_rows, n_cols)

    # Total grains
    total = np.sum(board)
    print(f"Total number of grains on {n_rows}x{n_cols} board: {total}")

    # Column averages
    col_avg = np.mean(board, axis=0)
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(1, n_cols + 1), col_avg)
    plt.xlabel("Column")
    plt.ylabel("Average grains")
    plt.title(f"Average grains per column ({n_rows}x{n_cols} board)")
    plt.show()

    # Heatmap
    plt.figure(figsize=(6, 6))
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title(f"Heatmap of grains on {n_rows}x{n_cols} board")
    plt.pcolor(board, cmap="viridis")
    plt.colorbar(label="Grains of wheat")
    plt.show()

    # First vs second half
    half = n_rows // 2
    first_half = np.sum(board[:half, :])
    second_half = np.sum(board[half:, :])
    ratio = second_half / first_half
    print(f"Second half / First half = {ratio:.2f}")

if __name__ == "__main__":
    analyze_board(8, 8)
# %%
