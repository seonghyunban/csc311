import numpy as np
import matplotlib.pyplot as plt

# Dimensions to evaluate (powers of 2)
dimensions = [2**i for i in range(11)]  # [2^0, 2^1, ..., 2^10]
num_points = 100  # number of points to sample in the unit cube
num_pairs = num_points * (num_points - 1) // 2  # total number of pairs of points

# Initialize lists to store the averages and standard deviations
mean_squared_euclidean = []
std_squared_euclidean = []
mean_l1 = []
std_l1 = []

# Loop over each dimension
for d in dimensions:
    # Sample 100 points in the unit cube [0, 1]^d
    points = np.random.rand(num_points, d)

    # Calculate all pairs of distances
    squared_euclidean_distances = []
    l1_distances = []

    for i in range(num_points):
        for j in range(i+1, num_points):
            diff = points[i] - points[j]

            # Squared Euclidean distance
            squared_euclidean_distances.append(np.sum(diff**2))

            # L1 distance (Manhattan distance)
            l1_distances.append(np.sum(np.abs(diff)))

    # Record statistics for squared Euclidean distances
    mean_squared_euclidean.append(np.mean(squared_euclidean_distances))
    std_squared_euclidean.append(np.std(squared_euclidean_distances))

    # Record statistics for L1 distances
    mean_l1.append(np.mean(l1_distances))
    std_l1.append(np.std(l1_distances))

# Plot results
plt.figure(figsize=(12, 6))

# Plot average distances
plt.subplot(1, 2, 1)
plt.plot(dimensions, mean_squared_euclidean, label='Mean Squared Euclidean Distance', marker='o')
plt.plot(dimensions, mean_l1, label='Mean L1 Distance', marker='s')
plt.xlabel('Dimension (d)')
plt.ylabel('Average Distance')
plt.title('Average Distance vs Dimension')
plt.legend()

# Plot standard deviations
plt.subplot(1, 2, 2)
plt.plot(dimensions, std_squared_euclidean, label='Std Dev Squared Euclidean Distance', marker='o')
plt.plot(dimensions, std_l1, label='Std Dev L1 Distance', marker='s')
plt.xlabel('Dimension (d)')
plt.ylabel('Standard Deviation of Distance')
plt.title('Standard Deviation vs Dimension')
plt.legend()

plt.tight_layout()
plt.show()
