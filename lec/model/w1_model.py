# Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d

# Create 2D dataset
X, y = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)
colors = ['red', 'green', 'blue','orange']

# Visualize the data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

for i in range(3):
    ax.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=f'Class {i}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)

ax.legend()
plt.title('2D Classification Dataset')
plt.show()


# Define distance function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Implement nearest neighbor function
def nearest_neighbor(xq, X):
    N = len(X)
    dv = []
    midx = 0
    mindist = float('inf')

    for i in range(N):
        di = euclidean_distance(xq, X[i])
        dv.append(di)

    for i in range(N):
        if dv[i] < mindist:
            midx = i
            mindist = dv[i]

    return midx, dv[midx]

# Function to plot nearest neighbor
def plot_nearest_neighbor(X, y, xq):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for i in range(3):
        ax.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=f'Class {i}')

    ax.scatter(xq[0], xq[1], c='black', s=200, marker='*', label='Query Point')

    nn_idx, _ = nearest_neighbor(xq, X)
    ax.scatter(X[nn_idx, 0], X[nn_idx, 1], c='black', s=100, marker='o', label='Nearest Neighbor')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.title('Nearest Neighbor Visualization')
    plt.show()


# Test nearest neighbor with different query points
query_points = [
    [0, 0],
    [5, 5],
    [-5, -5]
]

for xq in query_points:
    plot_nearest_neighbor(X, y, xq)


# Create and plot Voronoi diagram (using 2D projection for simplicity)
vor = Voronoi(X)
fig = plt.figure(figsize=(10, 8))
voronoi_plot_2d(vor)
plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in y])
plt.title('Voronoi Diagram')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
