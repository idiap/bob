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

def array_getitem(self, key):
  """Retrieves a single element or a slice from the array. This method follows
  the pythonic model for retrieving elements from mutable iterables. You can 
  also deploy negative keys or slices (e.g. '0:')."""
  if isinstance(key, (int,long)) or \
      (isinstance(key, tuple) and isinstance(key[0], (int,long))):
      return self.__getitem_internal__(key)
  return self.__getslice_internal__(key)

def array_setitem(self, key, value):
  """Sets a single element or a slice at the array. This method follows
  the pythonic model for setting elements at mutable iterables. You can also 
  deploy negative keys or slices (e.g. '0:')."""
  if isinstance(key, (int,long)) or \
      (isinstance(key, tuple) and isinstance(key[0], (int,long))):
      return self.__setitem_internal__(key, value)
  return self.__setslice_internal__(key, value)

for tname, atype in get_array_types():
  atype.__getitem__ = array_getitem
  atype.__setitem__ = array_setitem
  atype.__repr__ = atype.__str__

class __BlitzArrayTypeTester__(object):
  """A tester for blitz::Array<> types."""

  def __init__(self):
    self.types = tuple([k[1] for k in get_array_types()])

  def __call__(self, item):
    return isinstance(item, self.types)

is_blitz_array = __BlitzArrayTypeTester__()
