# Colorname recognizer for segmented objects

This directory contains a simple visual object recognition pipeline
for segmented objects. It contains a test dataset of ten objects, each
with about twenty images for training/validation.

The recognizer is based on a Naive Bayes Nearest Neighbor classifier
similar to that proposed in:

> Boiman, Oren, Eli Shechtman, and Michal Irani. "In defense of
> nearest-neighbor based image classification." CVPR 2008.

over local histograms of colorname descriptors as described in this paper:

> Van De Weijer, Joost, Cordelia Schmid, Jakob Verbeek, and Diane
> Larlus. "Learning color names for real-world applications." IEEE
> Transactions on Image Processing, 2009.

## Running the pipeline

The simplest way to run the pipeline with default parameters is to
first train the recognizer with:

`./train_recognizer.py dataset_jordi/`

and then run the validation script (which performs leave-one-out and
random split cross validation to estimate the accuracy of the trained recognizer):
 
`./validate_recognizer.py dataset_jordi/`

If you like, you can also run the simple example script which runs the
trained recognizer on a held-out test image:

`./run_recognizer.py dataset_jordi/`

See the scripts themselves for documentation on parameters and other
options.
