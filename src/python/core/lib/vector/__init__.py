from libpytorch_core_vector import *
__all__ = dir()

# adjustments to the __getitem__/__setitem__ mess
def get_vector_types():
  import inspect
  import libpytorch_core_vector

  def is_vector(t):
    if not inspect.isclass(t): return False
    cparts = t.__name__
    if cparts[:3] in ('boo','int','uin','flo','com'): return True
    return False

  return inspect.getmembers(libpytorch_core_vector, is_vector)

class __StdVectorTypeTester__(object):
  """A tester for std::vector<> types."""

  def __init__(self):
    self.types = tuple([k[1] for k in get_vector_types()])

  def __call__(self, item):
    return isinstance(item, self.types)

is_std_vector = __StdVectorTypeTester__()
del __StdVectorTypeTester__

def vec_str(self):
  """Nicely formats a std::vector<>"""
  return '[' + ', '.join([str(k) for k in self]) + ']'

def vec_repr(self):
  """Nicely formats the representation of a std::vector<>"""
  return "%s[%d] (0x%x)" % (self.__class__.__name__, len(self), id(self)) 

def vec_eq(self, other):
  if len(self) != len(other): return False
  for i, k in enumerate(self): 
    if k != other[i]: return False
  return True

def vec_ne(self, other):
  return not (self == other)

for vector_class in [k[1] for k in get_vector_types()]:
  vector_class.__str__ = vec_str
  vector_class.__repr__ = vec_repr
  vector_class.__eq__ = vec_eq
  vector_class.__ne__ = vec_ne

del vec_str
del vec_repr
del vec_eq
del vec_ne
