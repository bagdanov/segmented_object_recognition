import sys
import argparse
import pylab as pl
from utils import Timer
from CNRecognizer import CNRecognizer
from PythonCN import extract_all_image_features
from NBNN import NBNN

# Parse command line arguments.
parser = argparse.ArgumentParser(description='Example of how to run trained CNRecognizer.')
parser.add_argument('datadir', metavar='datadir', type=str,
                    help='dataset directory')
args = parser.parse_args()

# Read images.
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

