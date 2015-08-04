import numpy as np
from sklearn.neighbors import NearestNeighbors

#
# Simple class implementing Naive Bayes Nearest Neighbor
# classification algorithm. The default algorithm uses local features
# to vote for classes. The Boiman variant from this paper:
#
#  Boiman, Oren, Eli Shechtman, and Michal Irani. "In defense of
#  nearest-neighbor based image classification." CVPR 2008.
#
# is also imlemented (use the 'boiman=True' argument to the
# constructor).
#
# Andrew D. Bagdanov
# 27/07/2015
#
class NBNN:
    # Constructor:
    #  boiman  - whether to use the Boiman NBNN variant.
    #  verbose - spits out diagnostic information.
    def __init__(self, verbose=False):
        self.verbose_ = verbose

    # Fits the classifier to given data X (rows are local features)
    # and y (class labels).
    def fit(self, X, y):
        self.y_ = y
        self.classes_ = np.unique(y)
        self.kdtree_ = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
        if self.verbose_:
            print 'Fitting kd-tree to training data.'
        self.kdtree_.fit(X)

    # And predict the class of an input set of local descriptors.
    def predict(self, X, boiman=False):
        (distances, indices) = self.kdtree_.kneighbors(X, 2)
        class_indices = np.asarray(self.y_[indices[:, 0]].ravel(), dtype=np.uint8)
        min_dist = 1000000000.0
        best = -1
        if boiman:
            import ipdb; ipdb.set_trace()
            for cls in np.unique(class_indices):
                dist = np.sum(distances[np.where(class_indices == cls), 0])
                if dist < min_dist:
                    min_dist = dist
                    best = cls
            return best
        else:
            passed = np.where((distances[:, 1] / (distances[:, 0] + 0.00001)) >= 1.5)
            indices = indices[passed]
            class_votes = np.bincount(class_indices, minlength=len(self.classes_))
            probs = np.asarray(class_votes, dtype=np.float64)
            return probs / (probs.sum() + 0.000001)

