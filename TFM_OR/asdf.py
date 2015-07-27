import glob
import pylab as pl
import numpy as np

Folder = glob.glob("C:\Users\ACER\Desktop\TFM_OR\TFM_OR\dataset_jordi\*")
Dir = '%s\RGB\*.png' % (Folder[0])
Imgs = glob.glob(Dir)
print(Imgs[1])

img = pl.imread('car.png')
pl.imshow(img)
pl.show()