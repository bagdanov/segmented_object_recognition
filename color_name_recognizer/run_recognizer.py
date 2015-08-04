import sys
import argparse
import num
import pylab as pl
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
#clf.train(10, 5, 4)
if not clf._trained:
    print 'CNRecognizer in {} not trained! See train_recognizer.py script.'
    sys.exit(1)

# Extract masked features from all images.
#(Xtr, ytr) = extract_all_image_features(clf._dataset,
#                                        n_jobs=4,
#                                        patch_size=10,
#                                        stride=5)

# And classify.
pred = clf.predict(image, mask)


