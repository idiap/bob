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

class __BlitzArrayTypeTester__(object):
  """A tester for blitz::Array<> types."""

  def __init__(self):
    self.types = tuple([k[1] for k in get_array_types()])

  def __call__(self, item):
    return isinstance(item, self.types)

is_blitz_array = __BlitzArrayTypeTester__()
del __BlitzArrayTypeTester__

# binds string and representation
def array_str(self):
  """String representation. Used when printing or string conversion."""
  return "%s" % self.as_ndarray()
def array_repr(self):
  """Simplified string representation."""
  return "%s %s (0x%x)" % (self.cxx_blitz_typename, self.shape(), id(self)) 
for array_class in [k[1] for k in get_array_types()]:
  array_class.__str__ = array_str
  array_class.__repr__ = array_repr
del array_str
del array_repr

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
