#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 28 Oct 08:41:11 2011 

"""Arrayset additions
"""

from libpytorch_io import Arrayset

def arrayset_iter(self):
  """Allows Arraysets to be iterated in native python"""
  n = 0
  while n != len(self):
    yield self[n]
    n += 1
  raise StopIteration
Arrayset.__iter__ = arrayset_iter
del arrayset_iter

def arrayset_extend(self, obj, dim=0):
  """Extends the current Arrayset by either slicing the given array and
  appending each individual slice or iterating over an iterable containing
  arrays.

  Keyword Parameters:

  obj
    The object to extend this Arrayset with, may be arrays, io.Array's
    or an iterable full of arrays.

  dim
    **Iff** the input object is a single array, you will be able to specificy
    along which dimension such array should be sliced using this parameter. The
    value of `dim` is otherwise ignored.
  """

  if isinstance(obj, (tuple,list)):
    return self.__iterable_extend__(obj)

  else: #it is an io.Array or a ndarray
    if not isinstance(obj, Array): obj = Array(obj) #try cast
    return self.__array_extend__(obj, dim)

Arrayset.extend = arrayset_extend
del arrayset_extend

def arrayset_repr(self):
  """A simple representation"""
  return '<Arrayset[%d] %s@%s>' % (len(self), self.type.dtype, self.type.shape)
Arrayset.__repr__ = arrayset_repr
Arrayset.__str__ = arrayset_repr
del arrayset_repr

def arrayset_eq(self, other):
  """Compares two arraysets for content equality."""
  if self.shape != other.shape: return False 
  if len(self) != len(other): return False
  for i in range(len(self)):
    if self[i] != other[i]: return False
  return True
Arrayset.__eq__ = arrayset_eq
del arrayset_eq

def arrayset_ne(self, other):
  """Compares two arraysets for content inequality."""
  return not (self == other)
Arrayset.__ne__ = arrayset_ne
del arrayset_ne

def arrayset_cat(self, firstDim=False):
  """Concatenates an entire Arrayset in a single array.

  The original arrays will be organized by creating as many entries as
  necessary in the last dimension of the resulting array. If the option
  'firstDim' is set to True, then the first dimension of the resulting array is
  used for the disposal of the input arrays. 

  In this way, to retrieve the first array of the arrayset from the resulting
  array, you must either use:

  .. code-block:: python

    bz = arrayset.cat() #suppose an arrayset with shape == (4,) (1D array)
    bz[:,0] #retrieves the first array of the set if firstDim is False
    bz[0,:] #retrieves the first array of the set if firstDim is True

  The same is valid for N (N>1) dimensional arrays.

  Note this will load all the arrayset data in memory if that is not already
  the case and will copy all the data once (to the resulting array).

  .. warning::
    This method will only work as long as the resulting array number of
    dimensions is supported. Currently this means that self.shape has to have
    length 3 or less. If the Array data is 4D, the resulting ndarray would
    have to be 5D and that is not currently supported.
  """
  ashape = self.shape
  retshape = list(ashape)

  if firstDim: #first dimension contains examples
    retshape.insert(0, len(self))
    retval = numpy.zeros(retshape, dtype=self.elementType.name)

    for i, k in enumerate(self): #fill
      add = tuple([i] + [slice(d) for d in ashape])
      retval[add] = self[i].get()

  else: #last dimension contains examples
    retshape.append(len(self))
    retval = numpy.ndarray(retshape, dtype=self.elementType.name)

    for i, k in enumerate(self): #fill
      add = tuple([slice(d) for d in ashape] + [i])
      retval[add] = self[i].get()

  return retval
Arrayset.cat = arrayset_cat
del arrayset_cat

def arrayset_foreach(self, meth):
  """Applies a transformation to the Arrayset data by passing every
  array to the given method
  
  .. note::

    This will trigger loading all data elements within the Array and will
    create copies of the array data (that is returned).
  """
  return self.__class__([meth(k.get()) for k in self])
Arrayset.foreach = arrayset_foreach
del arrayset_foreach



