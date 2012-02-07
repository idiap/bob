#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 28 Oct 08:41:11 2011 

"""Arrayset additions
"""

from libpybob_io import Arrayset
import numpy

def arrayset_iter(self):
  """Allows Arraysets to be iterated in native python"""
  n = 0
  while n != len(self):
    yield self[n]
    n += 1
  raise StopIteration
Arrayset.__iter__ = arrayset_iter

def arrayset_repr(self):
  """A simple representation"""
  return '<Arrayset[%d] %s@%s>' % (len(self), self.type.dtype, self.type.shape)
Arrayset.__repr__ = arrayset_repr
Arrayset.__str__ = arrayset_repr

def arrayset_eq(self, other):
  """Compares two arraysets for content equality."""
  if self.shape != other.shape: return False 
  if len(self) != len(other): return False
  for i in range(len(self)):
    if not numpy.array_equal(self[i], other[i]): return False
  return True
Arrayset.__eq__ = arrayset_eq

def arrayset_ne(self, other):
  """Compares two arraysets for content inequality."""
  return not (self == other)
Arrayset.__ne__ = arrayset_ne

def arrayset_foreach(self, meth):
  """Applies a transformation to the Arrayset data by passing every
  array to the given method
  
  .. note::

    This will trigger loading all data elements within the Array and will
    create copies of the array data (that is returned).
  """
  return self.__class__([meth(k) for k in self])
Arrayset.foreach = arrayset_foreach

# Here are some legacy methods to keep up with the api changes
def arrayset_shape(self):
  return self.type.shape
Arrayset.shape = property(arrayset_shape)

def arrayset_eltype(self):
  return self.type.cxxtype
Arrayset.elementType = property(arrayset_eltype)

def arrayset_dtype(self):
  return self.type.dtype
Arrayset.dtype = property(arrayset_dtype)
