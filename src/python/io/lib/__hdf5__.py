#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 28 Oct 08:59:10 2011 

"""HDF5 additions
"""

from libpytorch_io import HDF5Type, HDF5File

# Some HDF5 addons
def hdf5type_str(self):
  return "%s@%s" % (self.type_str(), self.shape())
HDF5Type.__str__ = hdf5type_str
del hdf5type_str

def hdf5type_repr(self):
  return "<HDF5Type: %s (0x%x)>" % (str(self), id(self))
HDF5Type.__repr__ = hdf5type_repr
del hdf5type_repr

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
