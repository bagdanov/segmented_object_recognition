import os
import sys
import utils
import argparse
import numpy as np
from joblib import Parallel, delayed
from dataset import Dataset
from CNRecognizer import CNRecognizer
from PythonCN import extract_all_image_features
from NBNN import NBNN

##############################################################################
# Leave-one-out (LOO) and random split cross validation support.
def mulzip(lol, l):
    ret = []
    return np.hstack(n * np.ones((len(x),)) for (x, n) in zip(lol, l))
        
def holdout_split(I, Xtr, ytr):
    Xtr_t = utils.allbut(Xtr, I)
    ytr_t = mulzip(Xtr_t, utils.allbut(ytr, I))
    Xtr_t = np.vstack(Xtr_t)
    clf = NBNN()
    clf.fit(Xtr_t, ytr_t)
    return [clf.predict(Xtr[i]).argmax() for i in I]

def do_loo_crossvalidation(Xtr, ytr):
    print '\nPerforming leave-one-out crossvalidation on training set.'
    preds = Parallel(n_jobs=args.n_jobs)(delayed(holdout_split)([i], Xtr, ytr) for i in range(len(Xtr)))
    acc_loo = np.sum(np.asarray(sum(preds, []) == ytr) / float(len(Xtr)))
    print 'LOO crossvalidation accuracy: {}'.format(acc_loo)

def do_random_crossvalidation(Xtr, ytr):
    num_test = int(np.floor(len(Xtr) * args.splitprop))
    trials = args.trials
    acc_split = []
    print '\nPerforming {} splits of {} train and {} test.'.format(trials, len(Xtr) - num_test, num_test)
    splits = [np.random.permutation(range(len(Xtr)))[:num_test] for i in range(trials)]
    preds = Parallel(n_jobs=args.n_jobs)(delayed(holdout_split)(I, Xtr, ytr) for I in splits)
    acc_split = [np.sum(p == ytr[I]) / float(num_test) for (p, I) in zip(preds, splits)]
    print 'Accuracy @ {}% test: {}'.format(args.splitprop * 100, np.mean(acc_split))

##############################################################################
# Main script

# Setup argument parser.
parser = argparse.ArgumentParser(description='Validate Naive Bayes Nearest Neighbor classifier using LOO and random splits.')
parser.add_argument('--n_jobs', metavar='n_jobs', type=int,
                    default=10, help='number of parallel jobs for feature extraction')
parser.add_argument('--trials', metavar='trials', type=int,
                    default=10, help='number of trials for random split crossvalidation')
parser.add_argument('--splitprop', metavar='splitprop', type=float,
                    default=0.5, help='proportion of data to withold as training')
parser.add_argument('datadir', metavar='datadir', type=str,
                    help='dataset directory')

# Parse command line arguments.
args = parser.parse_args()

# Load model.
print 'Loading model...'
clf = CNRecognizer(args.datadir)
print 'Done.'

# Make sure model is trained.
if not clf._trained:
    print 'Model not trained! See train_model.py for model training.'
    sys.exit(1)

# Load the dataset for feature extraction.
dataset = Dataset(args.datadir + '/train')

# Extract masked features from all images.
(Xtr, ytr) = extract_all_image_features(dataset,
                                        n_jobs=args.n_jobs,
                                        patch_size=clf._patch_size,
                                        stride=clf._stride)
# Do LOO cross validation.
do_loo_crossvalidation(Xtr, ytr)

# And also random split cross validation.
do_random_crossvalidation(Xtr, ytr)
