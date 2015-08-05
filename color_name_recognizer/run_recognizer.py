#!/usr/bin/env python
#
# This is a very simple script that shows how to run a trained
# classifier on test images. Run it like this:
#
#  ./run_recognizer.py dataset_jordi/
#
# The script will instantiate and deserialize a trained classifier
# from the provided <datadir> directory, then use it to predict the
# class of a test image (given by the 'rubiks_asus_test_image.png' and
# 'rubiks_asus_test_mask.png' images). It prints some timing and
# prediction information.
#
# See:
#  train_recognizer.py - for a script that trains a recognizer on a dataset.
#
# TODO: Make the timing information more statistically meaningful.
# TODO: Switch to using PIL for image I/O.
#

import sys
import argparse
import pylab as pl
from utils import Timer
from CNRecognizer import CNRecognizer

# Parse command line arguments.
parser = argparse.ArgumentParser(description='Example of how to run trained CNRecognizer.')
parser.add_argument('datadir', metavar='datadir', type=str,
                    help='dataset directory')
args = parser.parse_args()

# Read images. NOTE WELL: image channels *MUST* be between 0 and
# 255. For some goddamn reason pylab scales PNGs to [0,1], but not
# JPEGs. Thanks. Anyway, be careful.
image = pl.imread('rubiks_asus_test_image.png') * 255.0
mask = pl.imread('rubiks_asus_test_mask.png')

# Instantiate (and deserialize, if already trained) recognizer.
clf = CNRecognizer(args.datadir)
if not clf._trained:
    print 'CNRecognizer in {} not trained! See train_recognizer.py script.'
    sys.exit(1)

# And classify.
with Timer('time to extract and classify'):
    pred = clf.predict(image, mask)

# Use the '_dataset' attribute of the recognizer to access classes.
print 'Class probabilities:'
for (label, p) in enumerate(pred):
    print '  {0:.2f}: {1:s}'.format(p, clf._dataset.label2class(label))
print '\nPrediction: {}'.format(clf._dataset.label2class(pl.argmax(pred)))

