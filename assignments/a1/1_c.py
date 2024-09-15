import numpy as np
import matplotlib.pyplot as plt
import random
from tools import euclidean_distance, manhattan_distance

d = list(range(11))
dimensions = [2 ** i for i in d]
sample_size = 100

eu_dist_mean = []
man_dist_mean = []
eu_dist_stdev = []
man_dist_stdev = []

for d in dimensions:
     sample = np.random.rand(sample_size, d)
     eu_dist = []
     man_dist = []

     for i in range(sample_size):
          for j in range(i, sample_size):
               eu_dist.append(euclidean_distance(sample[i], sample[j]))
               man_dist.append(manhattan_distance(sample[i], sample[j]))

     eu_dist_mean.append(np.mean(eu_dist))
     eu_dist_stdev.append(np.std(eu_dist))

     man_dist_mean.append(np.mean(man_dist))
     man_dist_stdev.append(np.std(man_dist))


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(dimensions, eu_dist_mean, label="l1 distance", marker='o')
plt.plot(dimensions, man_dist_mean, label="l2 distance", marker='s')
plt.xlabel("Dimension")
plt.ylabel("Average Distance")
plt.title("Dimension vs Mean Distance Plot")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(dimensions, eu_dist_stdev, label="l1 distance", marker='o')
plt.plot(dimensions, man_dist_stdev, label='l2 distance', marker='s')
plt.xlabel("Dimension")
plt.ylabel("Standard Deviation")
plt.title("Dimension vs Standard Deviation Plot")
plt.legend()

plt.tight_layout()
plt.show()




