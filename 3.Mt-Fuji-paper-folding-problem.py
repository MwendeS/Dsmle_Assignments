# %%
import math
# Problem 1: Minimum folds for Mount Fuji
def folds_to_exceed_height(height, t0=0.00008):
    """
    Calculate the minimum number of folds needed to exceed a given height.
    Parameters
    ----------
    height : float
        Target height in meters
    t0 : float
        Initial paper thickness (meters), default 0.00008
    Returns
    -------
    n_folds : int
        Minimum number of folds required
    """
    n = 0
    thickness = t0
    while thickness < height:
        n += 1
        thickness = t0 * (2 ** n)
    return n

# Problem 2: Examples
fuji_height = 3776  # meters
moon_distance = 384_400_000  # meters
proxima_distance = 4.0175e16  # meters

fuji_folds = folds_to_exceed_height(fuji_height)
moon_folds = folds_to_exceed_height(moon_distance)
proxima_folds = folds_to_exceed_height(proxima_distance)

print(f"Mount Fuji requires {fuji_folds} folds.")
print(f"Moon requires {moon_folds} folds.")
print(f"Proxima Centauri requires {proxima_folds} folds.")

# Problem 3: Length of paper needed
def paper_length_for_folds(n, t0=0.00008):
    """
    Calculate the length of paper needed to fold n times.
    
    Formula: L = (π * t0 / 6) * (2^n + 4)(2^n - 1)
    """
    return (math.pi * t0 / 6) * ((2 ** n) + 4) * ((2 ** n) - 1)
# Lengths required for Fuji, Moon, and Proxima
L_fuji = paper_length_for_folds(fuji_folds)
L_moon = paper_length_for_folds(moon_folds)
L_proxima = paper_length_for_folds(proxima_folds)
print(f"Length needed for Fuji folds: {L_fuji:.2e} meters")
print(f"Length needed for Moon folds: {L_moon:.2e} meters")
print(f"Length needed for Proxima folds: {L_proxima:.2e} meters")

# %%
