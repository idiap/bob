from libpytorch_database import *
import os

def dataset_arrayset_index(self):
  """Returns a dictionary containing the arrayset-id (key) and the arrayset
  itself (value)."""
  retval = {}
  for k, v in zip(self.ids(), self.arraysets()): retval[k] = v
  return retval
Dataset.arraysetIndex = dataset_arrayset_index

def dataset_relationset_index(self):
  """Returns a dictionary containing the relationset-name (key) and the 
  relationset itself (value)."""
  retval = {}
  for k, v in zip(self.names(), self.relationsets()): retval[k] = v
  return retval
Dataset.relationsetIndex = dataset_relationset_index

def dataset_eq(self, other):
  """Compares two datasets for equality, by comparing if their arraysets and
  relationsets are numerically equal. Please note that this method will defer
  the comparision of arraysets and relationsets to their respective __eq__()
  operators. Things like name and version will be skipped."""
  if sorted(self.ids()) != sorted(other.ids()): return False
  if sorted(self.names()) != sorted(other.names()): return False
  for id in self.ids(): 
    if self[id] != other[id]: return False
  for name in self.names():
    if self[name] != other[name]: return False
  return True
Dataset.__eq__ = dataset_eq

def dataset_ne(self, other):
  return not (self == other)
Dataset.__ne__ = dataset_ne

def arrayset_array_index(self):
  """Returns a dictionary containing the array ids (keys) and the arrays
  themselves (values)."""
  retval = {}
  for k in self.ids(): retval[k] = self[k]
  return retval
Arrayset.arrayIndex = arrayset_array_index

def arrayset_append(self, *args):
  import numpy
  from .. import core
  if len(args) == 1:
    if isinstance(args[0], Array):
      return self.__append_array__(args[0])
    elif core.array.is_blitz_array(args[0]):
      return self.__append_array__(Array(args[0]))
    elif isinstance(args[0], (str, unicode)):
      return self.__append_array__(Array(args[0]))
    else:
      raise RuntimeError, "Can only append database::Array, blitz::Array or filename to Arrayset"
  elif len(args) == 2:
    if isinstance(args[0], (str, unicode)) and instance(args[1], str):
      return self.__append_array__(args[0], args[1])
    else: 
      raise RuntimeError, "Can only append (filename,codecname) to Arrayset"
  raise RuntimeError, "This cannot happen!"

Arrayset.append = arrayset_append

def arrayset_setitem(self, id, *args):
  import numpy
  from .. import core
  if len(args) == 1:
    if isinstance(args[0], Array):
      self.__setitem_array__(id, args[0])
      return
    elif core.array.is_blitz_array(args[0]):
      self.__setitem_array__(id, Array(args[0]))
      return
    elif isinstance(args[0], (str, unicode)):
      self.__setitem_array__(id, Array(args[0]))
      return
    else:
      raise RuntimeError, "Can only set database::Array, blitz::Array or filename to Arrayset"
  elif len(args) == 2:
    if isinstance(args[0], (str, unicode)) and instance(args[1], str):
      self.__setitem_array__(id, Array(args[0], args[1]))
      return
    else: 
      raise RuntimeError, "Can only set (filename,codecname) to Arrayset"
  raise RuntimeError, "This cannot happen!"

Arrayset.append = arrayset_append

def arrayset_eq(self, other):
  """Compares two arraysets for content equality. We don't compare roles!"""
  if self.shape != other.shape: return False 
  #if self.role != other.role: return False
  if len(self) != len(other): return False
  for id in self.ids():
    if not other.exists(id): return False
    if self[id] != other[id]: return False
  return True
Arrayset.__eq__ = arrayset_eq

def arrayset_ne(self, other):
  """Compares two arraysets for content inequality. We don't compare roles!"""
  return not (self == other)
Arrayset.__ne__ = arrayset_ne

def array_get(self):
  """Returns a blitz::Array object with the internal element type"""
  return getattr(self, '__get_%s_%d__' % (self.elementType.name, len(self.shape)))()
Array.get = array_get

def array_cast(self, eltype):
  """Returns a blitz::Array object with the required element type"""
  return getattr(self, '__cast_%s_%d__' % (eltype.name, len(self.shape)))()
Array.cast = array_cast

def array_copy(self):
  """Returns a blitz::Array object which is a copy of the internal data"""
  return getattr(self, '__cast_%s_%d__' % (self.elementType.name, len(self.shape)))()
Array.copy = array_copy

def array_eq(self, other):
  """Compares two arrays for numeric equality"""
  return (self.get() == other.get()).all()
Array.__eq__ = array_eq

def array_ne(self, other):
  """Compares two arrays for numeric equality"""
  return not (self == other)
Array.__ne__ = array_ne

def relationset_index(self):
  """Returns a standard python dictionary that contains as keys, the roles and
  as values, python tuples containing the members (tuples) associated with each 
  role inside every member. Here is an example:
  
  { #roles      #members 1st. rel.  #members 2nd. relation  #... etc.
    'id':       1,                  2,                      ...)
    'pattern': ((member1, member2), (member3, member4),     ...)
    'target' : ((member101,),       (member102,), ...)
  }

  Please note that a member is actually a tuple with two values indicating the
  arrayset-id and the array-id that this member actually points to. If you want
  to get hold of a similar dictionary with members replaced by arrays and
  arraysets, please use the Dataset.relationsetIndex(relationset_name) method.
  """

  retval = {}
  retval['__id__'] = self.ids()
  roles = self.roles()
  for role in roles: retval[role] = [] #initialization
  for k in self.ids():
    for memberRole, members in self.memberDict(k).iteritems():
      retval[memberRole].append(members) #only works if initialized!
  for role in roles: retval[role] = tuple(retval[role]) #make it read-only
  return retval

Relationset.index = relationset_index

def relationset_eq(self, other):
  """Compares the contents of two relationsets to see if they match"""
  if sorted(self.roles()) != sorted(other.roles()): return False
  for k in self.roles():
    if self[k] != other[k]: return False
  if sorted(self.ids()) != sorted(other.ids()): return False
  for k in self.ids():
    if self[k] != other[k]: return False
  return True
Relationset.__eq__ = relationset_eq

def relationset_ne(self, other):
  return not (self == other)
Relationset.__ne__ = relationset_ne

def dataset_relationset_index_by_name(self, name):
  """Returns a dictionary like the one in Relationset.index(), but replaces the
  member tuples with real arrays or arraysets, as requested.
  """
  def map_one(dataset, member):
    """Maps one member as either an arrayset or array"""
    if member[1]: return dataset[member[0]][member[1]]
    else: return dataset[member[0]]

  retval = self[name].index()
  for role in [k for k in retval.keys() if k != '__id__']:
    replace_role = []
    for member_tuple in retval[role]: 
      replace_role.append(tuple([map_one(self, m) for m in member_tuple]))
    retval[role] = tuple(replace_role)
  return retval

Dataset.relationsetIndexByName = dataset_relationset_index_by_name

def binfile_getitem(self, i):
  """Returns a blitz::Array<> object with the expected element type and shape"""
  return getattr(self, '__getitem_%s_%d__' % \
      (self.elementType.name, len(self.shape)))(i)

BinFile.__getitem__ = binfile_getitem

def relation_eq(self, other):
  return sorted(self.members()) == sorted(other.members())
Relation.__eq__ = relation_eq

def relation_ne(self, other):
  return not (self == other)
Relation.__ne__ = relation_ne

def rule_eq(self, other):
  return (self.min == other.min) and (self.max == other.max)
Rule.__eq__ = rule_eq

def rule_ne(self, other):
  return not (self == other)
Rule.__ne__ = rule_ne

# Some HDF5 addons
def hdf5type_array_class(self):
  """Returns the array class in torch.core.array that is good type for me"""
  from ..core import array
  return getattr(array, '%s_%d' % (self.type_str(), len(self.shape())))
HDF5Type.array_class = hdf5type_array_class

def hdf5file_read(self, path, pos=-1):
  """Reads elements from the current file.
  
  Parameters:
  path -- This is the path to the HDF5 dataset to read data from
  pos -- This is the position in the dataset to readout. If the given value is
  smaller than zero, we read all positions in the dataset and return you a
  list. If the position is specific, we return a single element.
  """
  dtype = self.describe(path)
  if dtype.is_array():
    if pos < 0: # read all
      return [self.read(path, k) for k in range(self.size(path))]
    else:
      retval = dtype.array_class()(dtype.shape())
      self.__read_array__(path, pos, retval)
      return retval
  else:
    if pos < 0: # read all
      return [self.read(path, k) for k in range(self.size(path))]
    else:
      return getattr(self, '__read_%s__' % dtype.type_str())(path, pos)
HDF5File.read = hdf5file_read

def hdf5file_append(self, path, data, dtype=None):
  """Appends data to a certain HDF5 dataset in this file.

  Parameters:
  path -- This is the path to the HDF5 dataset to append data to
  data -- This is the data that will be appended. If this element is an
  interable element (list or tuple), we will append all elements in such
  iterable.
  dtype -- Is an optional parameter that forces the conversion from the type
  given in 'data' to one of the supported torch element types. Please note that
  the data has to be convertible to the given type by means of boost::python
  otherwise an error is raised. Also note this has no effect in case data are
  composed of arrays (in which case the selection is automatic).
  """
  from ..core import array

  def best_type (value):
    """Returns the approximate best type for a given python value"""
    if isinstance(value, bool): return 'bool'
    elif isinstance(value, int): return 'int32'
    elif isinstance(value, long): return 'int64'
    elif isinstance(value, float): return 'float64'
    elif isinstance(value, complex): return 'complex128'
    elif isinstance(value, (str, unicode)): return 'string'
    return 'UNSUPPORTED'
  
  if not isinstance(data, (list, tuple)): data = [data]

  if array.is_blitz_array(data[0]):
    for k in data: self.__append_array__(path, k)

  else: #is scalar, in which case the user may have given a dtype
    if dtype is None: dtype = best_type(data[0])
    meth = getattr(self, '__append_%s__' % dtype)
    for k in data: meth(path, k)
HDF5File.append = hdf5file_append

def hdf5file_replace(self, path, pos, data, dtype=None):
  """Replaces data to a certain HDF5 dataset in this file.

  Parameters:
  path -- This is the path to the HDF5 dataset to append data to
  pos -- This is the position we should replace
  data -- This is the data that will be appended. 
  dtype -- Is an optional parameter that forces the conversion from the type
  given in 'data' to one of the supported torch element types. Please note that
  the data has to be convertible to the given type by means of boost::python
  otherwise an error is raised. Also note this has no effect in case data are
  composed of arrays (in which case the selection is automatic).
  """
  from ..core import array

  def best_type (value):
    """Returns the approximate best type for a given python value"""
    if isinstance(value, bool): return 'bool'
    elif isinstance(value, int): return 'int32'
    elif isinstance(value, long): return 'int64'
    elif isinstance(value, float): return 'float64'
    elif isinstance(value, complex): return 'complex128'
    elif isinstance(value, (str, unicode)): return 'string'
    return 'UNSUPPORTED'
  
  if array.is_blitz_array(data):
    for k in data: self.__replace_array__(path, pos, k)

  else: #is scalar, in which case the user may have given a dtype
    if dtype is None: dtype = best_type(data[0])
    meth = getattr(self, '__replace_%s__' % dtype)
    for k in data: meth(path, pos, k)
HDF5File.append = hdf5file_append
