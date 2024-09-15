import numpy as np
import matplotlib.pyplot as plt

# This imports the Seaborn library, which is built on top of Matplotlib
# and provides a high-level interface for making attractive and
# informative statistical graphics.
import seaborn as sns


# It changes the appearance of all the following plots to make them more
# aesthetically pleasing by applying Seaborn's default settings
# (such as plot colors, font sizes, and layout spacing).
sns.set() #???


# NOTE: Decision Trees
# Let's construct and fit a simple decision tree on 2-dimensional data.
# Each sample has one of four possible labels, denoted by a different color
# in the visualization below.
from sklearn.datasets import make_blobs

# * make_blobs generates a synthetic dataset with 300 samples.
# * n_samples=300 specifies the total number of samples.
# * centers=4 means the data will be generated with 4 distinct clusters
# (each representing a different label).
# * random_state=0 ensures reproducibility by setting a seed
# for the random number generator.
# * cluster_std=1.0 sets the standard deviation of the clusters,
# controlling their spread.
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0) #???

# * plt.scatter creates a scatter plot.
# * X[:, 0] and X[:, 1] are the x and y coordinates of the data points, respectively.
# * c=y sets the color of each point according to its label y.
# * Different labels will be shown in different colors.
# * s=50 specifies the size of the scatter plot markers.
# * cmap='rainbow' uses the 'rainbow' colormap to assign colors to
# different labels.
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow') #???


# NOTE: Model fitting and visualization
# We use the `DecisionTreeClassifier` from Scikit-Learn to fit this data
# and visualize its predictions.
from sklearn.tree import DecisionTreeClassifier


# DecisionTreeClassifier(criterion='entropy') initializes a decision tree
# classifier using the entropy criterion for splitting.
# .fit(X, y) trains the decision tree on the dataset X (features) and y (labels)
tree = DecisionTreeClassifier(criterion='entropy').fit(X, y)


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    # code adapted from this handbook:
    # https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook

    # ax = ax or plt.gca(): If an ax object (axis) is provided, it will be used.
    # Otherwise, it will use the current active axis (plt.gca()). This allows
    # the function to either use an existing axis or create a new one.
    ax = ax or plt.gca() #???

    # Plot the training points
    # ax.scatter(...): Plots the training points with colors representing
    # their labels.
    # c=y: Colors points based on their labels.
    # s=30: Sets the size of the markers.
    # cmap=cmap: Uses the specified colormap for coloring.
    # clim=(y.min(), y.max()): Sets the color limits for consistent coloring.
    # zorder=3: Sets the drawing order to ensure points are on top of other
    # plot elements.
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3) #???

    # ax.axis('tight'): Adjusts the axis limits to fit the data tightly.
    ax.axis('tight') #???
    # ax.axis('off'): Hides the axis lines and labels for a cleaner visualization.
    ax.axis('off') #???

    # xlim and ylim: Capture the current x and y axis limits to use for
    # plotting the decision boundaries.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y) #???

    # np.meshgrid: Creates a grid of points covering the x and y axis ranges.
    # The grid is used to evaluate the classifier’s predictions across the
    # entire plotting area.
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200)) #???

    # np.c_[xx.ravel(), yy.ravel()]: Stacks the grid coordinates into a
    # 2D array suitable for predictions.
    # model.predict(...): Predicts the class for each grid point.
    # .reshape(xx.shape): Reshapes the predictions to match the grid shape
    # for plotting.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) #???

    # Create a color plot with the results
    # ax.contourf(...): Plots the decision boundaries using filled contour plots.
    # levels=np.arange(n_classes + 1) - 0.5: Defines the contour levels to
    # separate different classes.
    # alpha=0.3: Sets the transparency of the contours.
    # cmap=cmap: Uses the specified colormap.
    # zorder=1: Sets the drawing order to ensure contours are behind the
    # scatter points.
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap,
                           zorder=1) #???

    # ax.set(...): Restores the axis limits to match the previously captured limits.
    ax.set(xlim=xlim, ylim=ylim) #???

visualize_classifier(tree, X, y)

# NOTE: Random Forests through Bagging
# Recall the technique of Bagging, using which we create an ensemble of Decision Trees
# (called Random Forest) to reduce overfitting.
# Below, we create a Random Forest of 100 Decision Tree estimators,
# where 80% of the data is randomly sampled with replacement to fit each estimator.

from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(criterion='entropy')

# BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1)
# Creates a Bagging classifier using the decision tree as the base estimator.
# tree: The base estimator (decision tree) for the Bagging ensemble.
# n_estimators=100: Specifies the number of decision trees in the ensemble
# (100 trees).
# max_samples=0.8: Indicates that each decision tree is trained on 80% of
# the data sampled with replacement.
# random_state=1: Ensures reproducibility by setting a random seed.

bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,
                        random_state=1) #???

# bag.fit(X, y): Trains the Bagging classifier on the dataset X (features)
# and y (labels).
bag.fit(X, y) #???
visualize_classifier(bag, X, y) #???


# NOTE: Bagging reduces Variance
# We saw in lecture that bagging reduces variance. Let's check this empirically.
# Below, we perform bagging with different number of estimators and compute
# the empirical variance in their predictions.

# np.arange(30, 101, step=10): Creates an array of numbers from 30 to
# 100 (inclusive) with a step of 10. This array represents different numbers
# of estimators (decision trees) to use in Bagging.
ns = np.arange(30, 101, step=10) #???

variance = []
for n in ns:
    # tree = DecisionTreeClassifier(criterion='entropy'): Initializes a
    # decision tree classifier.
    # bag = BaggingClassifier(tree, n_estimators=n, max_samples=0.8,
    # random_state=1): Initializes a Bagging classifier with n decision trees.
    # bag.fit(X, y): Trains the Bagging classifier on the dataset.

    tree = DecisionTreeClassifier(criterion='entropy')
    bag = BaggingClassifier(tree, n_estimators=n, max_samples=0.8,
                            random_state=1)
    bag.fit(X, y)
    # for tr in bag.estimators_:: Loops over each trained decision tree
    # in the Bagging ensemble.
    # predictions.append(tr.predict(X)): Appends the predictions from each
    # decision tree to the predictions list.
    predictions = []
    for tr in bag.estimators_: #???
        predictions.append(tr.predict(X)) #???
    predictions = np.array(predictions)
    # var = np.var(predictions, axis=1): Computes the variance of predictions
    # for each sample across the different decision trees.
    # variance.append(np.mean(var)): Computes the mean variance of predictions
    # across all samples and appends it to the variance list.
    var = np.var(predictions, axis=1)
    variance.append(np.mean(var))

plt.figure()
plt.plot(ns, variance)
plt.xlabel("Number of trees in the forest")
plt.ylabel("Variance of the estimator")
plt.show()
