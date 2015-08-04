import os
import numpy as np
from utils import pickle, unpickle
from dataset import Dataset
from PythonCN import extract_all_class_features
from NBNN import NBNN

class CNRecognizer:
    def __init__(self, datadir, modelname='CNRecognizer.pkl'):
        self._datadir = datadir
        self._modelname = modelname
        self._trained = False
        if os.path.exists(datadir + '/models/' + modelname):
            inst = unpickle(datadir + '/models/' + modelname)
            for att in dir(inst):
                setattr(self, att, getattr(inst, att))

    def train(self, patch_size, stride, n_jobs):
        self._patch_size = patch_size
        self._stride = stride

        # Scan for all image files in dataset training dir.
        dataset = Dataset(self._datadir + '/train')

        # Extract masked features from all training images.
        (Xtr, ytr) = extract_all_class_features(dataset,
                                                n_jobs=n_jobs,
                                                patch_size=patch_size,
                                                stride=stride)

        # Stack image features and labels into arrays (they were lists).
        Xtr = np.vstack(Xtr)
        ytr = np.hstack(ytr)
        
        # Make a Naive Bayes NN classifier and train it.
        self._clf = NBNN()
        self._clf.fit(Xtr, ytr)
        self._trained = True

    def serialize(self):
        pickle(self, self._datadir + '/models/' + self._modelname)

        
