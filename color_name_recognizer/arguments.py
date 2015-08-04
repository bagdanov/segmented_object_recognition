import argparse

# Setup argument parser.
parser = argparse.ArgumentParser(description='Train Naive Bayes Nearest Neighbor classifier on color name histograms.')
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
