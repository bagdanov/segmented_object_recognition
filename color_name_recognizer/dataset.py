from os.path import basename
from glob import glob

class Dataset:
    """Simple class wrapping a dataset of images and segmentation masks.

    Args:
      directory (string): root directory with class subdirs

    Notes:
      Assumes dataset directory is organized according to
      convention. Level-1 subdirectories are classnames (the directory
      name is used as the classname). Tn these class directories
      should be two directories, RGB and Mask, with contain the
      corresponding images and segmentation masks.
    """
    def __init__(self, directory):
        self._directory = directory
        self._classes = map(basename, sorted(glob(directory + '/*')))
        self._class2label = dict((y, x) for (x, y) in (enumerate(self._classes)))
        self._imagenames = {}
        for cls in self._classes:
            images = sorted(glob(self._directory + '/' + cls + '/RGB/*.png'))
            masks = sorted(glob(self._directory + '/' + cls + '/Mask/*.png'))
            self._imagenames[cls] = zip(images, masks)
        
    def get_class_images(self, cls):
        """Retrieve images corresponding to classname.

        Args:
          cls (string): classname to retrieve images for.

        Returns: 
          list of (im, mask) filename tuples
        """
        return iter(self._imagenames[cls])

    def label2class(self, label):
        """Converts integer :label: to classname string."""
        return self._classes[label]
        
    def class2label(self, cls):
        """Converts classname string to integer label."""
        return self._class2label[cls]
    
    @property
    def classes(self):
        """Returns the list of classnames in dataset."""
        return self._classes

            
