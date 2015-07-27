import glob
import numpy as np
import pylab as pl
from PythonCN import gridCN
from sklearn.neighbors import NearestNeighbors

Folder = glob.glob("C:\\Users\\ACER\\Desktop\\TFM_OR\\TFM_OR\\dataset_jordi\\*")
all_features = []
GT = []
counter = 0
step_size = 5
feature_size = 5

for i in range(0, len(Folder)-1):
    Dir = "%s\\RGB\\*.png" %(Folder[i])
    Dir2 = "%s\\Mask\\*.png" %(Folder[i])
    Imgs = glob.glob(Dir)
    Msks = glob.glob(Dir2)
    for j in range(0, len(Imgs)-1):
        counter = counter + 1
        Img = pl.imread(Imgs[j]) * 255.0
        Mask = pl.imread(Imgs[j]) * 255.0
        pl.imshow(Img)
        pl.show()

        # Generate a dense grid over the image (stride 4).
        grid = np.meshgrid(np.arange(0, Img.shape[1], 4), np.arange(0, Img.shape[0], 4), indexing='ij')
        grid = zip(grid[0].ravel(), grid[1].ravel())

        # And extract hard assigned names.
        hard_cns = gridCN(Img, grid, 5, hard_assign=True)

        # And extract soft assigned names.
        soft_cns = gridCN(Img, grid, 5, hard_assign=False)
        
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
