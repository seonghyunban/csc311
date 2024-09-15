import numpy as np

def euclidean_distance(A, B):
    return np.sum((A - B) ** 2)

def manhattan_distance(A, B):
    return np.sum(np.abs(A - B))

