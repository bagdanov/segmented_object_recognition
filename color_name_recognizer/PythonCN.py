import numpy as np
import scipy.io as sio
import pylab as pl
from joblib import Parallel, delayed

# Load the w2m color mapping matrix.
w2c = sio.loadmat('w2c.mat')['w2c']
w2c_argmax = np.argmax(w2c, 1)

# Function to convert RGB image to hard or soft CN map.
def im2c(im, hard_assign=True):
    RR = im[:, :, 0].ravel()
    GG = im[:, :, 1].ravel()
    BB = im[:, :, 2].ravel()
    index_im = np.floor(RR / 8) + 32 * np.floor(GG / 8) + 32 * 32 * np.floor(BB / 8)
    index_im = np.array(index_im, dtype=np.uint32)
    if hard_assign:
        return w2c_argmax[index_im].reshape(im.shape[:2])
    else:
        return w2c[index_im].reshape(im.shape[:2] + (11,))

# Function to pool CN map over grid of patches.
def gridCN(im, grid, patch_size, hard_assign=True):
    # Compute the color name mapping at all pixels.
    cns = im2c(im, hard_assign=hard_assign)

    # Half the patch size.
    half = int(np.floor(patch_size / 2))

    # Process each grid point in sequence.
    patches = []
    for (x, y) in grid:
        # No fancy border handling.
        lx = max(0, x - half)
        ux = min(im.shape[1], x + half + 1)
        ly = max(0, y - half)
        uy = min(im.shape[0], y + half + 1)
        patch = cns[ly:uy, lx:ux]
        if hard_assign:
            patches.append(np.bincount(patch.ravel(), minlength=11))
        else:
            patches.append(patch.sum(0).sum(0) / np.prod(patch.shape[:2]))

    # Normalize patches if hard assigning.
    patches = np.asarray(patches, dtype=np.float32)
    return patches / patches.sum(1)[:, None] if hard_assign else patches

def extract_masked_cns(im, mask, stride=5, patch_size=10):
    """Extracts local color name descriptors from image foreground."""
    # Check if images passed as fnames (or ndarrays).
    if isinstance(im, str):
        im = pl.imread(im) * 255.0
    if isinstance(mask, str):
        mask = pl.imread(mask)

    # Make the local patch grid, mask, and make (x, y) pairs.
    grid = np.meshgrid(np.arange(0, im.shape[1], stride),
                       np.arange(0, im.shape[0], stride),
                       indexing='ij')
    I = np.where(mask[grid[1].ravel(), grid[0].ravel()])[0]
    grid = zip(grid[0].ravel()[I], grid[1].ravel()[I])

    # And return the local CN histograms.
    return gridCN(im, grid, patch_size, hard_assign=False)

# Extract masked features.
def extract_all_image_features(dataset, n_jobs=1, stride=5, patch_size=10):
    """Extract masked features from all dataset images, return features and labels"""
    cns = []
    labels = []
    for (label, cls) in enumerate(dataset.classes):
        print 'Extracting masked CNs from class {}'.format(cls)
        hists = Parallel(n_jobs=n_jobs)(delayed(extract_masked_cns)(imname, maskname) for (imname, maskname) in dataset.get_class_images(cls))
        #        hists = np.vstack(hists)
        labels.append(label * np.ones((len(hists),), dtype=np.float32))
        cns.append(hists)
    
    # Stack lists in numpy arrays.
    return (sum(cns, []), np.hstack(labels))

# Extract masked features.
def extract_all_class_features(dataset, n_jobs=1, stride=5, patch_size=10):
    """Extract masked features from all dataset images, return features and labels"""
    cns = []
    labels = []
    for (label, cls) in enumerate(dataset.classes):
        print 'Extracting masked CNs from class {}'.format(cls)
        hists = Parallel(n_jobs=n_jobs)(delayed(extract_masked_cns)(imname, maskname) for (imname, maskname) in dataset.get_class_images(cls))
        hists = np.vstack(hists)
        labels.append(label * np.ones((len(hists),), dtype=np.float32))
        cns.append(hists.astype(np.float32))
    
    # Stack lists in numpy arrays.
    return (cns, labels)
