import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d

# Create 2D dataset, sample size 300, 2 features, 3 categories, seed = 42.
X, Y = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)
colors = ['red', 'green', 'blue']

# Create Scatter plot
# Creates a new figure object with a specified size of 10 inches by 8 inches.
fig = plt.figure(figsize=(10, 8))

# Adds a subplot to the figure with a grid layout of 1x1 (only one subplot).
# The 111 means "1 row, 1 column, 1st subplot".
ax = fig.add_subplot(111)


for i in range(3):
    ax.scatter(X[Y == i, 0], X[Y == i, 1], c=colors[i], label=f"Class {i}")

# Set axis label and limit
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-15, 15)
plt.ylim(-15, 15)

# Set title
plt.title("2D Classification Dataset")

# Show legend and show the plot
plt.legend()
plt.show()


# Define Euclidean distance function
def EuDist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Define Nearest neighbor function
def NN(xq, x):
    dists = []
    for xd in x:
        dists.append(EuDist(xq, xd))

    min_x = 0
    min_dist = dists[0]

    for i in range(1, len(dists)):
        if dists[i] < min_dist:
            min_dist = dists[i]
            min_x = i

    return min_x


# Function that plots the nearest neighbor
def plot_nearest_neighbor(x, y, xq):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for i in range(3):
        ax.scatter(x[y == i, 0], x[y == i, 1], c=colors[i], label=f"Class{i}")

    ax.scatter(xq[0], xq[1], c="black", marker="*", s=200, label="Query Point")
    min_x = NN(xq, x)
    ax.scatter(x[min_x, 0], x[min_x, 1] , c="black", marker="o", s=100, label="Nearest Neighbor")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    plt.legend()
    plt.title("Nearest Neighbor Visualization")
    plt.show()


# Create some query points and plot the nearest points
query_points = [
     [0, 0],
     [5, 5],
     [-5, -5]
]

for xq in query_points:
    plot_nearest_neighbor(X, Y, xq)

# Create and plot voronoi diagram
vor = Voronoi(X)
voronoi_plot_2d(vor)
plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in Y])
plt.title('Voronoi Diagram')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()








