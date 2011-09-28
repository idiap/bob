from libpytorch_io import *
import numpy
import os

# Attach loading and saving functionality to arrays
def load(filename, codecname=''):
  """Loads an array from a given file path specified
  
  Parameters:
  filename -- (string) The path to the file from which an array will be loaded
  in memory.
  codecname -- (string, optional) The name of the Array codec that will be used
  to decode the information in the file. If not provided, I will try to derive
  the codec to be used from the filename extension.
  """
  return Array(filename, codecname=codecname).get()

def save(obj, filename, codecname=''):
  """Saves the current array at the file path specified.

  Parameters:
  filename -- (string) The path to the file in which this array will be saved
  to.
  codecname -- (string, optional) The name of the Array codec that will be
  used. If not provided, I will try to derive the codec to be used from the
  filename extension.
  """
  Array(obj).save(filename, codecname=codecname)

def arrayset_iter(self):
  """Allows Arraysets to be iterated in native python"""
  n = 0
  while n != len(self):
    yield self[n]
    n += 1
  raise StopIteration
Arrayset.__iter__ = arrayset_iter
del arrayset_iter

def arrayset_append(self, *args):
  if len(args) == 1:
    if isinstance(args[0], Array):
      self.__append_array__(args[0])
      return
    elif isinstance(args[0], numpy.ndarray):
      self.__append_array__(Array(args[0]))
      return
    elif isinstance(args[0], (str, unicode)):
      self.__append_array__(Array(args[0]))
      return
    else:
      raise RuntimeError, "Can only append io::Array, array or filename to Arrayset"
  elif len(args) == 2:
    if isinstance(args[0], (str, unicode)) and instance(args[1], str):
      self.__append_array__(args[0], args[1])
      return
    else: 
      raise RuntimeError, "Can only append (filename,codecname) to Arrayset"
  raise RuntimeError, "This cannot happen!"

Arrayset.append = arrayset_append
del arrayset_append

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

  if hasattr(obj, '__iter__'):
    return self.__iterable_extend__(obj)

  else: #it is an io.Array or a ndarray
    if not isinstance(obj, Array): obj = Array(obj) #try cast
    return self.__array_extend__(obj, dim)

Arrayset.extend = arrayset_extend
del arrayset_extend

def arrayset_setitem(self, id, *args):
  if len(args) == 1:
    if isinstance(args[0], Array):
      self.__setitem_array__(id, args[0])
      return
    elif isinstance(args[0], numpy.ndarray):
      self.__setitem_array__(id, Array(args[0]))
      return
    elif isinstance(args[0], (str, unicode)):
      self.__setitem_array__(id, Array(args[0]))
      return
    else:
      raise RuntimeError, "Can only set io::Array, array or filename to Arrayset"
  elif len(args) == 2:
    if isinstance(args[0], (str, unicode)) and instance(args[1], str):
      self.__setitem_array__(id, Array(args[0], args[1]))
      return
    else: 
      raise RuntimeError, "Can only set (filename,codecname) to Arrayset"
  raise RuntimeError, "This cannot happen!"

Arrayset.__setitem__ = arrayset_setitem
del arrayset_setitem

def arrayset_repr(self):
  """A simple representation"""
  return '<Arrayset[%d] %s@%s>' % (len(self), self.elementType, self.shape)
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
    retval = numpy.ndarray(dtype=self.elementType, shape=retshape)

    for i, k in enumerate(self): #fill
      add = tuple([i] + [slice(d) for d in ashape])
      retval[add] = self[i].get()

  else: #last dimension contains examples
    retshape.append(len(self))
    retval = numpy.ndarray(dtype=self.elementType, shape=retshape)

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

def array_get(self):
  """Returns a array object with the internal element type"""
  return getattr(self, '__get_%s_%d__' % (self.elementType.name, len(self.shape)))()
Array.get = array_get
del array_get

def array_cast(self, eltype):
  """Returns a array object with the required element type"""
  return getattr(self, '__cast_%s_%d__' % (eltype.name, len(self.shape)))()
Array.cast = array_cast
del array_cast

def array_copy(self):
  """Returns a array object which is a copy of the internal data"""
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
  return "<Array %s@%s>" % (self.elementType, self.shape)
Array.__repr__ = array_repr
del array_repr

def array_str(self):
  """A complete representation for arrays"""
  return str(self.get())
Array.__str__ = array_str
del array_str

def binfile_getitem(self, i):
  """Returns a array<> object with the expected element type and shape"""
  return getattr(self, '__getitem_%s_%d__' % \
      (self.elementType.name, len(self.shape)))(i)

BinFile.__getitem__ = binfile_getitem
del binfile_getitem

def tensorfile_getitem(self, i):
  """Returns a array<> object with the expected element type and shape"""
  return getattr(self, '__getitem_%s_%d__' % \
      (self.elementType.name, len(self.shape)))(i)

TensorFile.__getitem__ = tensorfile_getitem
del tensorfile_getitem

# Some HDF5 addons
def hdf5type_str(self):
  return "%s@%s" % (self.type_str(), self.shape())
HDF5Type.__str__ = hdf5type_str
del hdf5type_str

def hdf5type_repr(self):
  return "<HDF5Type: %s (0x%x)>" % (str(self), id(self))
HDF5Type.__repr__ = hdf5type_repr
del hdf5type_repr

def hdf5file_read(self, path):
  """Reads all dataset elements from the current file. In this mode, the
  dataset is considered to contain a single element that will be read entirely
  from the file into memory as a array.
  
  Keyword Parameters:

  path
    This is the path to the HDF5 dataset to read data from
  """
  descr = self.describe(path)[1].type

  if descr.shape() == (1,):  # read as scalar
    return getattr(self, '__read_%s__' % descr.type_str())(path, 0)

  else: # read as array
    retval = numpy.ndarray(dtype=descr.type_str(), shape=self.shape())
    getattr(self, '__read_%s_array__' % descr.type_str())(path, 0, retval)
    return retval

HDF5File.read = hdf5file_read
del hdf5file_read

def hdf5file_lread(self, path, pos=-1):
  """Reads elements from the indicated dataset considering the dataset first
  dimension contains the number of elements in a list and that the dataset was
  created with append() instead of set(). Elements read have N-1 dimensions,
  where N is the number of dimensions displayed at the dataset.
  
  Keyword Parameters:

  path
    This is the path to the HDF5 dataset to read data from

  pos
    This is the position in the dataset to readout. If the given value is
    smaller than zero, we read all positions in the dataset and return you a
    list. If the position is specific, we return a single element.
  """
  dtype = self.describe(path)[0]
  descr = dtype.type
  size = dtype.size
  
  def read_scalar_or_array(self, path, descr, pos):
    if descr.shape() == (1,):  # read as scalar
      return getattr(self, '__read_%s__' % descr.type_str())(path, pos)

    else: # read as array
      retval = numpy.ndarray(dtype=descr.type_str(), shape=descr.shape())
      getattr(self, '__read_%s_array__' % descr.type_str())(path, pos, retval)
      return retval

  if pos < 0: # read all -- recurse
    return [read_scalar_or_array(self, path, descr, k) for k in range(size)]

  else:
    return read_scalar_or_array(self, path, descr, pos)

HDF5File.lread = hdf5file_lread
del hdf5file_lread

def hdf5file_append(self, path, data, compression=0, dtype=None):
  """Appends data to a certain HDF5 dataset in this file. When you append data
  to a dataset and such dataset does not yet exist, it is created with an extra
  dimension to accomodate an unlimited number of appends. If you wish to have a
  single element of the type, you are better off using set() instead of this
  method.

  Parameters:

  path
    This is the path to the HDF5 dataset to append data to
  
  data
    This is the data that will be appended. If this element is an
    interable element (list or tuple), we will append all elements in such
    iterable.

  compression
    If the append instruction creates a new Dataset inside the file, the value
    given by this variable will set the compression level of the chunked
    dataset (every array in the dataset will be compressed individually).  It
    is not worth to compress very small arrays as there will be some gzip
    header overhead on the process. The default value of '0' should be OK for
    most small arrays. Note this setting has no effect if you are appending a
    scalar it **only works for arrays**.

  dtype
    Is an optional parameter that forces the conversion from the type given in
    'data' to one of the supported torch element types. Please note that the
    data has to be convertible to the given type by means of boost::python
    otherwise an error is raised. Also note this has no effect in case data are
    composed of arrays (in which case the selection is automatic).
  """

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

  if isinstance(data[0], numpy.ndarray):
    for k in data: self.__append_array__(path, k, compression)

  else: #is scalar, in which case the user may have given a dtype
    if dtype is None: dtype = best_type(data[0])
    meth = getattr(self, '__append_%s__' % dtype)
    for k in data: meth(path, k)

HDF5File.append = hdf5file_append
del hdf5file_append

def hdf5file_set(self, path, data, compression=0, dtype=None):
  """Sets the scalar or array at position 0 to the given value. This method is
  equivalent to checking if the scalar/array at position 0 exists and then
  replacing it. If the path does not exist, we create the new scalar/array.
  In the case the dataset does not exist, a new dataset is created to
  accomodated your input value. This dataset will not accept expansion and you
  will not be able to append() to it. If you wish it behaves like that, use
  append() instead of this method.

  Parameters:

  path
    This is the path to the HDF5 dataset to append data to
  
  data
    This is the data that will be set. If this element is an interable element
    (list or tuple), we will set all elements in such iterable.

  compression
    If the set instruction creates a new Dataset inside the file, the value
    given by this variable will set the compression level of the chunked
    dataset (every array in the dataset will be compressed individually).  It
    is not worth to compress very small arrays as there will be some gzip
    header overhead on the process. The default value of '0' should be OK for
    most small arrays. Note this setting has no effect if you are setting a
    scalar it **only works for arrays**.

  dtype
    Is an optional parameter that forces the conversion from the type given in
    'data' to one of the supported torch element types. Please note that the
    data has to be convertible to the given type by means of boost::python
    otherwise an error is raised. Also note this has no effect in case data are
    composed of arrays (in which case the selection is automatic).
  """

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

  if isinstance(data[0], numpy.ndarray):
    for k in data: self.__set_array__(path, k, compression)

  else: #is scalar, in which case the user may have given a dtype
    if dtype is None: dtype = best_type(data[0])
    meth = getattr(self, '__set_%s__' % dtype)
    for k in data: meth(path, k)

HDF5File.set = hdf5file_set
del hdf5file_set

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

  def best_type (value):
    """Returns the approximate best type for a given python value"""
    if isinstance(value, bool): return 'bool'
    elif isinstance(value, int): return 'int32'
    elif isinstance(value, long): return 'int64'
    elif isinstance(value, float): return 'float64'
    elif isinstance(value, complex): return 'complex128'
    elif isinstance(value, (str, unicode)): return 'string'
    return 'UNSUPPORTED'
  
  if isinstance(data, numpy.ndarray):
    for k in data: self.__replace_array__(path, pos, k)

  else: #is scalar, in which case the user may have given a dtype
    if dtype is None: dtype = best_type(data[0])
    meth = getattr(self, '__replace_%s__' % dtype)
    for k in data: meth(path, pos, k)
HDF5File.replace = hdf5file_replace
del hdf5file_replace

__all__ = dir()
