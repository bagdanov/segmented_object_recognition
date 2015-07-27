import numpy as np
import scipy.io as sio
import pylab as pl

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
    half = np.floor(patch_size / 2)

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
