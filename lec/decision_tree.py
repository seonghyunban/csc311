import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

# setup aesthetic
sb.set()


# generate dataset
X, Y = make_blobs(n_samples=300, centers=4, random_state=0)
# plot data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='rainbow')


# load the decision tree classifier, and set the criterion
tree = DecisionTreeClassifier(criterion='entropy')
# fit the dataset to the loaded classifier
tree.fit(X, Y)

# Visualizer of the decision boundaries of a classifier
# given the training data points.
def tree_visualizer(model, X, Y, ax=None, cmap='rainbow'):

    # fit the model in case it has not been yet
    model.fit(X, Y)

    # create plot
    plot = ax or plt.gca()

    # scatter the data points
    plot.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap=cmap)

    # hide the axis lines
    plot.axis('off')

    # get x and y range
    xlim = plot.get_xlim()
    ylim = plot.get_ylim()

    # get np array of the discretized ranges of x and y
    x_grid = np.linspace(*xlim, num=200)
    y_grid = np.linspace(*ylim, num=200)

    # create a mesh grid
    # xx: An array where each row is a copy of the x vector.
    # yy: An array where each column is a copy of the y vector.
    xx, yy = np.meshgrid(x_grid, y_grid)

    # flatten 2D arrays xx and yy into long 1D array
    xx_r = xx.ravel()
    yy_r = yy.ravel()

    # column-wise concatenation of the two flattened array -> 1D array
    prediction_set = np.c_[xx_r, yy_r]

    # make prediction and reshape it back to the 2D array
    prediction = model.predict(prediction_set).reshape(xx.shape)

    # fill with the contour of prediction on the xx and yy meshgrid
    num_classes = len(np.unique(Y))
    countour = plot.contourf(xx, yy, prediction,
                             alpha=0.5,
                             levels=np.arange(num_classes + 1) - 0.5,
                             cmap=cmap,
                             zorder=1,
                             )

tree_visualizer(tree, X, Y)







plt.show()
