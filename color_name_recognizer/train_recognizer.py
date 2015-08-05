#!/usr/bin/env python
#
# This is a simple script that instantiates, trains and serializes a
# colorname masked object recognizer on the images in the provided
# dataset directory.
#
# Run it like this:
#
#  ./train_recognizer.py --n_jobs=4 --patch_size=10 --stride=5  dataset_jordi/
#
# The trained model is serialized to the <datadir>/models directory.
#
# See:
#
#  CNRecognizer.py - for the recognizer logic itself.
#  dataset.py - for the expected layout of the <datadir> directory.
#

import argparse
from CNRecognizer import CNRecognizer

# Setup argument parser.
parser = argparse.ArgumentParser(description='Train CNRecognizer on training dataset.')
parser.add_argument('--patch_size', metavar='patch_size', type=int,
                    default=10, help='size of local patches')
parser.add_argument('--n_jobs', metavar='n_jobs', type=int,
                    default=10, help='number of parallel jobs for feature extraction')
parser.add_argument('--stride', metavar='stride', type=int,
                    default=5,
                    help='stride of sliding window for patch extraction')
parser.add_argument('datadir', metavar='datadir', type=str,
                    help='dataset directory')

# Parse command line arguments.
args = parser.parse_args()

# Train model (caching is done automatically).
print 'Training and serializing model...'
clf = CNRecognizer(args.datadir)
clf.train(args.patch_size, args.stride, args.n_jobs)
clf.serialize()
print 'Done.'
