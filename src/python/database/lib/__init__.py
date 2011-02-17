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
