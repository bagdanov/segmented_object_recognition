# Simple example of using the Python CN interface.
import numpy as np
import pylab as pl
from PythonCN import gridCN
from sklearn.neighbors import NearestNeighbors

# Load the test image.
im = pl.imread('car.png') * 255.0

# Generate a dense grid over the image (stride 4).
grid = np.meshgrid(np.arange(0, im.shape[1], 4), np.arange(0, im.shape[0], 4), indexing='ij')
grid = zip(grid[0].ravel(), grid[1].ravel())

# And extract hard assigned names.
hard_cns = gridCN(im, grid, 5, hard_assign=True)

# And extract soft assigned names.
soft_cns = gridCN(im, grid, 5, hard_assign=False)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(hard_cns)
distances, indices = nbrs.kneighbors(hard_cns)

# Print some indicative values.
print 'Hard/soft @ 10:'
print hard_cns[10]
print soft_cns[10]

print '\nHard/soft @ 20:'
print hard_cns[20]
print soft_cns[20]

print '\nHard/soft @ 4000:'
print hard_cns[4000]
print soft_cns[4000]
