from ._io import *

from . import __array__
from . import __file__
from . import __hdf5__
from . import __video__

def load(inputs):
  """Loads the contents of a file, an iterable of files, or an 
  iterable of bob.io.Array's into a numpy ndarray

  Parameters:

  inputs
    This might represent several different entities:\n
    1. The name of a file (full path) from where to load the data.
       In this case, this assumes that the file contains an array
       and returns a loaded numpy ndarray.
    2. An iterable of filenames to be loaded in memory. In this
       case, this would assume that each file contains a single
       1D sample or a set of 1D samples, load them in memory and
       concatenate them into a single and returned 2D numpy
       ndarray.
    3. An iterable of bob.io.Array. In this case, this would
       assume that each bob.io.Array contains a single 1D sample
       or a set of 1D samples, load them in memory if required
       and concatenate them into a single and returned 2D numpy
       ndarray.
    4. An iterable with mixed filenames and bob.io.Array. In
       this case, this would returned a 2D numpy ndarray, as
       described by points 2. and 3..
  """

  from collections import Iterable
  import numpy
  if isinstance(inputs, (str, unicode)):
    return Array(inputs).get()
  elif isinstance(inputs, Iterable):
    retval = []
    for obj in inputs:
      if isinstance(obj, (str, unicode)):
        retval.append(load(obj))
      elif isinstance(obj, Array):
        retval.append(obj.get())
      else:
        raise TypeError("Iterable contains an object which is not a filename nor a bob.io.Array.")
    return numpy.vstack(retval)
  else:
    raise TypeError("Unexpected input object. This function is expecting a filename, or an iterable of filenames and/or bob.io.Array's")

def merge(filenames):
  """Converts an iterable of files into an iterable over bob.io.Array's 

  Parameters:

  filenames
    This might represent:\n
    1. A single filename. In this case, an iterable with a single
       'external' bob.io.Array is returned.
    2. An iterable of filenames to be converted into an iterable
       of 'external' bob.io.Arrays's
  """

  from collections import Iterable
  if isinstance(filenames, (str, unicode)):
    return [Array(filenames)]
  elif isinstance(filenames, Iterable):
    retval = []
    for filename in filenames:
      retval.append(Array(filename))
    return retval
  else:
    raise TypeError("Unexpected input object. This function is expecting an iterable of filenames.")

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
