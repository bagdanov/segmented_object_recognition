#
# Simple class implementing Naive Bayes Nearest Neighbor
# classification algorithm. The default algorithm uses local features
# to vote for classes. Loosely base on:
#
#  Boiman, Oren, Eli Shechtman, and Michal Irani. "In defense of
#  nearest-neighbor based image classification." CVPR 2008.
#
# TODO: Docstrings for class and methods
#

import numpy as np
from sklearn.neighbors import NearestNeighbors

class NBNN:
    # Constructor:
    #  verbose - spits out diagnostic information.
    def __init__(self, verbose=False):
        self._verbose = verbose

    # Fits the classifier to given data X (rows are local features)
    # and y (class labels).
    def fit(self, X, y):
        self._y = y
        self._classes = np.unique(y)
        self._kdtree = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
        if self._verbose:
            print 'Fitting kd-tree to training data.'
        self._kdtree.fit(X)

    # And predict the class of an input set of local descriptors.
    def predict(self, X, boiman=False):
        (distances, indices) = self._kdtree.kneighbors(X, 2)
        class_indices = np.asarray(self._y[indices[:, 0]].ravel(), dtype=np.uint8)
        passed = np.where((distances[:, 1] / (distances[:, 0] + 0.00001)) >= 1.5)
        indices = indices[passed]
        class_votes = np.bincount(class_indices, minlength=len(self._classes))
        probs = np.asarray(class_votes, dtype=np.float64)
        return probs / (probs.sum() + 0.000001)

