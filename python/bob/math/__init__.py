from ._math import *

# add some default distance functions
import math
import numpy

def euclidean_distance(x1, x2):
  """Returns the Euclidean distance between the given vectors"""
  return math.sqrt(((x1-x2)*(x1-x2)).sum())


def normalized_scalar_product(x1, x2):
  """Computes the normalized scalar product (also known as the cosine between the two vectors). 
  This is a similarity measure, the results are in the range [-1,1], 
  or [0,1] when both vectors only have positive coefficients."""
  # Since this is actually a similarity function, we return its negative
  return numpy.dot(x1, x2) / math.sqrt(numpy.dot(x1,x1) * numpy.dot(x2,x2))


__all__ = dir()
