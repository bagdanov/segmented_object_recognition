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

## Components

This recognizer module consists of a number of files.

Python modules:
* `CNRecognizer.py`: a class implementing the recognizer logic
  (basically a wrapper around Dataset and NBNN functionalities).
* `NBNN.py`: a class which implements the Naive Bayes Nearest Neighbor
  classifier (learns a kd-tree over training data for fast nearest
  neighbor calculation, prediction done by voting).
* `utils.py`: random useful crap.
* `dataset.py`: wraps access to files and classes in a dataset directory.
* `PythonCN.py`: implements local colorname histogram feature
  extraction.

Executable scripts:
* `train_recognizer.py`: train a recognizer on provided dataset directory.
* `validate_recognizer.py`: leave-one-out and random split validation
  of trained classifier.
* `run_recognizer.py`: example script showing how to run trained classifier.

Other files:
* `w2c.mat`: matrix coding mapping of quantized RGB values to
  colornames.
* `rubiks_asus_test_mask.png`: test object mask image.
* `rubiks_asus_test_image.png`: test object RGB image.
