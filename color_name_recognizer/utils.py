import time
from itertools import izip_longest
from pickle import load, dump

def allbut(l, I):
    """Return all elements from list l except indexes in I"""
    return [x for (j, x) in enumerate(l) if j not in I]

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

# Timer class.
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

def pickle(obj, fname):
    """Helper function to pickle obj to file fname"""
    with open(fname, 'wb') as fn:
        dump(obj, fn)

def unpickle(fname):
    """Helper function to unpickle object from file fname"""
    with open(fname, 'rb') as fn:
        r = load(fn)
    return r

