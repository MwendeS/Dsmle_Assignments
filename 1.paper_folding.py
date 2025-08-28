# %%
# Paper Folding Assignment
# Author: SHARON ANN MWENDE

import time
import matplotlib.pyplot as plt
# Constants
THICKNESS = 0.00008 # thickness of paper in meters
FOLDS = 43           # number of folds
MOON_DISTANCE_KM = 384400
# 1. Using power operator (**)
start = time.time()
folded_power = THICKNESS * (2 ** FOLDS)
elapsed_power = time.time() - start
print("=== Power Operator Method ===")
print("Thickness (meters):", folded_power)
print("Thickness (km): {:.2f}".format(folded_power / 1000))
print("Reached the moon?", "Yes" if folded_power/1000 >= MOON_DISTANCE_KM else "No")
print("Execution time: {:.10f} seconds".format(elapsed_power))
print()
# %%
# 2. Using for loop
start = time.time()
folded_loop = THICKNESS
for i in range(FOLDS):
    folded_loop *= 2
elapsed_loop = time.time() - start

print("=== For Loop Method ===")
print("Thickness (meters):", folded_loop)
print("Thickness (km): {:.2f}".format(folded_loop / 1000))
print("Reached the moon?", "Yes" if folded_loop/1000 >= MOON_DISTANCE_KM else "No")
print("Execution time: {:.10f} seconds".format(elapsed_loop))
print()
# %%
# 3. Save thickness values in a list
thickness_list = []
current = THICKNESS
thickness_list.append(current)  # before folding (n=0)

for i in range(FOLDS):
    current *= 2
    thickness_list.append(current)
# Check list length (should be 44: fold 0 to 43)
print("Number of thickness values saved in list:", len(thickness_list))
print()
# %%
# 4. Visualization
plt.figure(figsize=(10,6))
# Customizations
plt.title("Thickness of Folded Paper", fontsize=16)
plt.xlabel("Number of Folds", fontsize=14)
plt.ylabel("Thickness [m]", fontsize=14)
plt.tick_params(labelsize=12)

plt.plot(thickness_list, color="red", linestyle="--", linewidth=2, label="Red dashed") # Plot 1: red dashed line
plt.plot(thickness_list, color="green", linestyle="-", linewidth=3, alpha=0.6, label="Green solid") # Plot 2: green solid line
plt.plot(thickness_list, color="blue", linestyle=":", linewidth=2, label="Blue dotted") # Plot 3: blue dotted line
plt.legend(fontsize=12) # Show legend
plt.show()              # Show graph
#%%