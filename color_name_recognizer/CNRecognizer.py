import os
import numpy as np
from utils import pickle, unpickle
from dataset import Dataset
from PythonCN import extract_all_class_features, extract_masked_cns
from NBNN import NBNN

class CNRecognizer:
    '''
    Simple class implementing a recognizer based on local colorname histograms.

    This recognizer uses a Naive Bayes Nearest Neighbor classifier
    over local colorname patches. It trains over all image/mask pairs
    in the train/ directory of the dataset given as input.

    Instantiate it like this:

      clf = CNRecognizer(datadir, modelname)

    where datasetdir is the directory of the dataset, and modelname is
    the name of the serlialized model saved to the datasetdir/models
    directory when the serialize() method is called.
    
    NOTES:
    ------
    The serialization code is hacky and untested. Treat it like
    running with scissors.    
    '''
    def __init__(self, datadir, modelname='CNRecognizer.pkl'):
        self._datadir = datadir
        self._modelname = modelname
        self._trained = False
        if os.path.exists(datadir + '/models/' + modelname):
            inst = unpickle(datadir + '/models/' + modelname)
            for att in dir(inst):
                setattr(self, att, getattr(inst, att))

    def train(self, patch_size, stride, n_jobs):
        '''Train the model on all images in the train/ directory of the dataset'''
        self._patch_size = patch_size
        self._stride = stride

        # Scan for all image files in dataset training dir.
        self._dataset = Dataset(self._datadir + '/train')

        # Extract masked features from all training images.
        (Xtr, ytr) = extract_all_class_features(self._dataset,
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

    def predict(self, image, mask):
        '''
        Method to classify an image based on masked colorname
        patches. Handles feature extraction and masking.

        IMPORTANT: image channels MUST range between 0 and 255 (as
        opposed to 0.0 and 1.0)
        '''
        cns = extract_masked_cns(image, mask, self._stride, self._patch_size)
        return self._clf.predict(cns)
        
    def serialize(self):
        '''Pickle the trained recognizer to the models/ directory of the dataset.'''
        pickle(self, self._datadir + '/models/' + self._modelname)
