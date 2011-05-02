from libpytorch_core_array import *
__all__ = dir()

# adjustments to the __getitem__/__setitem__ mess
def get_array_types():
  import inspect
  import libpytorch_core_array

  def is_array(t):
    if not inspect.isclass(t): return False
    cparts = t.__name__.split('_')
    if len(cparts) == 2 and \
        cparts[0][:3] in ('boo','int','uin','flo','com') and \
        int(cparts[1]) in (1,2,3,4): return True
    return False

  return inspect.getmembers(libpytorch_core_array, is_array)

def get_3d_array_types():
  import inspect
  import libpytorch_core_array

  def is_array(t):
    if not inspect.isclass(t): return False
    cparts = t.__name__.split('_')
    if len(cparts) == 2 and \
        cparts[0][:3] in ('boo','int','uin','flo','com') and \
        int(cparts[1]) in (3): return True
    return False

  return inspect.getmembers(libpytorch_core_array, is_array)

class __BlitzArrayTypeTester__(object):
  """A tester for blitz::Array<> types."""

  def __init__(self):
    self.types = tuple([k[1] for k in get_array_types()])

  def __call__(self, item):
    return isinstance(item, self.types)

is_blitz_array = __BlitzArrayTypeTester__()
del __BlitzArrayTypeTester__

# to create a similar tensor as before
def __sameAs__(self):
  return self.__class__(self.shape())

# binds string and representation
def array_str(self):
  """String representation. Used when printing or string conversion."""
  return "%s" % self.as_ndarray()

def array_repr(self):
  """Simplified string representation."""
  return "%s %s (0x%x)" % (self.cxx_blitz_typename, self.shape(), id(self)) 

def array_convert(self, dtype, dstRange=None, srcRange=None):
  """Function which allows to convert/rescale a blitz array of a given type
     into a blitz array of an other type. Typically, this can be used to rescale a
     16 bit precision grayscale image (2d array) into an 8 bit precision grayscale
     image.

     Paramters:
     dtype -- (string) Controls the output element type for the returned array
     dstRange -- (tuple) Determines the range to be deployed at the returned array
     srcRange -- (tuple) Determines the input range that will be used for the scaling

     Returns:
     A blitz::Array with the same shape as this one, but re-scaled and with its element
     type as indicated by the user.
  """

  if dstRange is None and srcRange is None:
    return getattr(self, '__convert_%s__' % dtype)()
  elif dstRange is None:
    return getattr(self, '__convert_%s__' % dtype)(destRange=dstRange)
  elif srcRange is None:
    return getattr(self, '__convert_%s__' % dtype)(sourceRange=srcRange)
  else:
    return getattr(self, '__convert_%s__' % dtype)(destRange=dstRange, sourceRange=srcRange)

def array_save(self, filename, codecname=''):
  """Saves the current array at the file path specified.

  Parameters:
  filename -- (string) The path to the file in which this array will be saved
  to.
  codecname -- (string, optional) The name of the Array codec that will be
  used. If not provided, I will try to derive the codec to be used from the
  filename extension.
  """
  from ...database import Array
  Array(self).save(filename, codecname=codecname)

for array_class in [k[1] for k in get_array_types()]:
  array_class.__str__ = array_str
  array_class.__repr__ = array_repr
  array_class.convert = array_convert
  array_class.save = array_save
  array_class.sameAs = __sameAs__
del array_str
del array_repr
del array_convert
del __sameAs__

def load(filename, codecname=''):
  """Loads an array from a given file path specified
  
  Parameters:
  filename -- (string) The path to the file from which an array will be loaded
  in memory.
  codecname -- (string, optional) The name of the Array codec that will be used
  to decode the information in the file. If not provided, I will try to derive
  the codec to be used from the filename extension.
  """
  from ...database import Array
  return Array(filename, codecname=codecname).get()

def array(data, dtype=None):
  """Creates a new blitz::Array<T,N> through numpy. 
  
  The dimensionality is extracted from the data. The data-type (dtype) is
  inferred from the data if not given a la numpy.

  This method is handy for python -> blitz array conversions and may be
  inefficient for production code. In the latter case, please directly use the
  specific array constructors. Example:

  >>> direct = torch.core.array.float32_2(iterable, shape)

  Please note that direct constructors for blitz arrays require a non-nested
  iterable and a shape, even in the single dimensional case.

  Parameters
  ----------
  data: array_like
    An array, any object exposing the array interface, an object whose
    __array__ method returns a numpy array, or any (nested) sequence. This
    includes existing blitz arrays.

  dtype: data-type as a string or as numpy.dtype, optional
    The desired data-type for the array. If not given, then the type will be
    determined as the minimum type required to hold the objects in the
    sequence. This argument cannot be used to downcast the array. For that,
    use .cast(t) method. Currently supported data-types are:
    
    * bool
    * int8, 16, 32 or 64
    * uint8, 16, 32 or 64
    * float32, 64 (or 128) **
    * complex64, 128 (or 256) **

  ** Support for float128 or complex256 is experimental and may be unstable.
  Not all methods may accept arrays of this type.
  """
  from numpy import array
  import libpytorch_core_array
  np = array(data, dtype)
  cls = getattr(libpytorch_core_array, '%s_%d' % (np.dtype.name, np.ndim))
  return cls(np)
