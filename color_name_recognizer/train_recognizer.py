import os
import utils
import arguments
from dataset import Dataset
from CNRecognizer import CNRecognizer
# from PythonCN import extract_all_features
from NBNN import NBNN

# Parse command line arguments.
args = arguments.args

# Train model (caching is done automatically).
print 'Training and serializing model...'
clf = CNRecognizer(args.datadir)
clf.train(args.patch_size, args.stride, args.n_jobs)
clf.serialize()
print 'Done.'
