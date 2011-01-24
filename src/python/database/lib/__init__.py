from libpytorch_database import *

# We attach some high-level functionality to this module:
def load(path):
  """Loads the dataset from an arbitrary file."""
  from libpytorch_database import __load_local_xml__
  return __load_local_xml__(path)

def loadString(data):
  """Loads the dataset from a string"""
  from libpytorch_database import __load_local_xml__
  from tempfile import mkstemp
  (fd, name) = mkstemp(prefix='torch_dataset_', suffix='.xml')
  f = fdopen(fd, 'wt')
  f.write(data)
  f.close()
  return __load_local_xml__(data)

def dataset_arrayset_index(self):
  """Returns a dictionary containing the arrayset-id (key) and the arrayset
  itself (value)."""
  retval = {}
  for k in self.arraysets: retval[k.id] = k
  return retval

def dataset_relationset_index(self):
  """Returns a dictionary containing the relationset-name (key) and the 
  relationset itself (value)."""
  retval = {}
  for k in self.relationsets: retval[k.name] = k
  return retval

Dataset.arraysetIndex = property(dataset_arrayset_index)
Dataset.relationsetIndex = property(dataset_relationset_index)

def arrayset_array_index(self):
  """Returns a dictionary containing the array ids (keys) and the arrays
  themselves (values)."""
  retval = {}
  for k in self.arrays: retval[k.id] = k
  return retval

Arrayset.arrayIndex = property(arrayset_array_index)

def array_copy(self):
  """Returns a blitz::Array object with the expected type and dimension"""
  from .. import core
  retval = getattr(core.array, '%s_%d' % \
      (self.getParentArrayset().arrayType.name, len(self.getParentArrayset().shape)))() #empty blitz::Array
  self.bzcopy(retval)
  return retval

def array_refer(self):
  """Returns a blitz::Array object with the expected type and dimension, which
  points to the internally allocated data."""
  from .. import core
  retval = getattr(core.array, '%s_%d' % \
      (self.getParentArrayset().arrayType.name, len(self.getParentArrayset().shape)))() #empty blitz::Array
  self.bzrefer(retval)
  return retval

Array.copy = array_copy
Array.refer = array_refer

def relationset_index(self):
  """Returns a standard python dictionary that contains as keys, the roles and
  as values, python tuples containing the Dataset::Members associated with each 
  role inside every member. Here is an example:
  
  { #roles      #members 1st. rel.  #members 2nd. relation  #... etc.
    '__id__':   1,                  2,                      ...)
    'pattern': ((member1, member2), (member3, member4),     ...)
    'target' : ((member101,),       (member102,), ...)
  }

  To extract blitz arrays from the returned value of this method you must
  iterate over each returned member and, from their 'arrays' method, choose
  to copy or refer to the array data in the order that suits you the best.
  """
  retval = {}
  retval['__id__'] = []
  roles = [k.role for k in self.rules]
  for role in roles: retval[role] = [] #initialization
  for k in self.relations:
    for role in roles: retval[role].append(k.membersWithRole(role))
    retval['__id__'].append(k.id)
  for role in roles: retval[role] = tuple(retval[role]) #make it read-only
  retval['__id__'] = tuple(retval['__id__']) 
  return retval

Relationset.index = property(relationset_index)

def member_arrays(self, arraysets):
  """Returns a tuple of arrays (containing one or more arrays) that were
  described in this relation member. Please note that by using this property
  you may trigger loading of data.

  N.B.: If the array-id == 0, the member points to an Arrayset.
  """
  tmp = [k for k in arraysets if k.id == self.arraysetId]
  if not tmp:
    raise RuntimeException, "Cannot find Arrayset with id=%d pointed by Member in Relationset" % self.arraysetId
  
  #if you find it , because of the schema restrictions, there has to be only 1
  tmp = tmp[0]
  if self.arrayId == 0: #we would like to have all arrays
    return tmp.arrays

  #if you get to this point, we are talking about a specific array
  tmp = [k for k in tmp.arrays if k.id == self.arrayId]
  if not tmp:
    raise RuntimeException, "Cannot find Array with id=%d in Arrayset with id=%d, pointed by Member in Relationset" % (self.arrayId, self.arraysetId)

  #if you find it , because of the schema restrictions, there has to be only 1
  return tuple(tmp)

Member.arrays = member_arrays
