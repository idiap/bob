from libpytorch_core_array import *
__all__ = dir()

import numpy
def as_torch(a):
  """Converts a numpy array into an equivalent blitz array, if that is
  possible (i.e. there must be a python-bound C++ equivalent for the input
  array scalar type."""
  exec('obj = %s(a)' % equivalent_scalar(a))
  return obj
