from ._cxx import *

from . import __arrayset__
from . import __array__
from . import __file__
from . import __hdf5__

def load(filename):
  """Loads the contents of a file into a numpy ndarray

  Parameters:

  filename
    The name of the file (full path) from where to load the data.
  
  """
  return Array(filename).get()

def save(array, filename, dtype=None):
  """Saves the contents of an array-like object to file
  
  Parameters:

  array
    The array-like object

  filename
    The name of the file where you need the contents saved to

  dtype
    A description type to coerce the input data to, if required.
  """
  return Array(array, dtype).save(filename)

__all__ = dir()
