#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 28 Oct 08:58:04 2011 

"""Array additions
"""

from ._cxx import Array

def array_cast(self, dtype):
  """Returns an array object with the required element type"""
  return self.get().astype(dtype)
Array.cast = array_cast
del array_cast

def array_copy(self):
  """Returns an array object which is a copy of the internal data"""
  return getattr(self, '__cast_%s_%d__' % (self.elementType.name, len(self.shape)))()
Array.copy = array_copy
del array_copy

def array_eq(self, other):
  """Compares two arrays for numeric equality"""
  return (self.get() == other.get()).all()
Array.__eq__ = array_eq
del array_eq

def array_ne(self, other):
  """Compares two arrays for numeric equality"""
  return not (self == other)
Array.__ne__ = array_ne
del array_ne

def array_repr(self):
  """A simple representation for generic Arrays"""
  return "<Array %s@%s>" % (self.type.dtype, self.type.shape)
Array.__repr__ = array_repr
del array_repr

def array_str(self):
  """A complete representation for arrays"""
  return str(self.get())
Array.__str__ = array_str
del array_str

# Here are some legacy methods to keep up with the api changes
def array_shape(self):
  return self.type.shape
Array.shape = property(array_shape)

def array_eltype(self):
  return self.type.cxxtype
Array.elementType = property(array_eltype)

def array_dtype(self):
  return self.type.dtype
Array.dtype = property(array_dtype)
