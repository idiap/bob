#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 11 Nov 10:38:52 2011 

"""Tests the io::Array interface from python.
"""

import os
import sys
import unittest
import torch
import numpy
import tempfile

def tempname(suffix, prefix='torchtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

class ArrayTest(unittest.TestCase):
  """Performs various tests for the Torch::io::Array objects"""

  def test01_init_onthefly(self):

    # You can initialize io.Array's with NumPy ndarrays:
    a1 = numpy.array(range(4), 'float32').reshape(2,2)
    A1 = torch.io.Array(a1)

    # Use the .get() operator to retrieve the data as a NumPy ndarray
    self.assertTrue( numpy.array_equal(a1, A1.get()) )

    # When you initialize an Array to an existing ndarray in contiguous memory,
    # then it actually captures a reference - it does not copy the data.
    a1[0,1] = 3.14
    self.assertTrue ( (A1.get()[0,1] - a1[0,1]) < 1e-6 )

    # Just to make the point, we delete the reference to a1 and A1, all works
    # as expected:
    a1_ref = A1.get()
    del a1
    del A1
    self.assertTrue ( (a1_ref[0,1] - 3.14) < 1e-6 )

    # Optionally, you can construct the Array using any array-like object:
    A2 = torch.io.Array([[3, 5], [7, 9]])

    # That, of course, creates a new reference and a copy to the data, creating
    # internally its own numpy.ndarray and storing that.
    self.assertEqual ( A2.get().dtype, numpy.dtype('int') )

    # The form for creating an Array as above does not say anything about the
    # data type. We can do it by specifying it. Anything that is acceptable by
    # numpy.dtype() can be put in there:
    A3 = torch.io.Array([[3, 5], [7, 9]], 'float64')
    self.assertEqual ( A3.get().dtype, numpy.dtype('float64') )

  def test02_init_fromfiles(self):

    # Initialization from files is possible using the io.File object.
    f = torch.io.open('test1.hdf5', 'r')
    A1 = torch.io.Array(f) # reads all contents.

    self.assertEqual ( A1.shape, (3, 3) )
    self.assertEqual ( A1.dtype, numpy.dtype('uint16') )
    self.assertTrue ( numpy.array_equal( A1.get()[1,:], [12, 19, 35] ) )

    # You can also read files in "Arrayset" mode, in which we discretize the
    # first dimension. For example, reading a file that contains a single 2D
    # array [1, 2], [3, 4] at position 1 will actually only extract the 1D
    # array [3, 4]. See this:

    A1 = torch.io.Array(f, 1)
    
    self.assertEqual ( A1.shape, (3,) )
    self.assertEqual ( A1.dtype, numpy.dtype('uint16') )
    self.assertTrue ( numpy.array_equal( A1.get(), [12, 19, 35] ) )

    # Initializing by giving a filename is also possible, it is the same as
    # just giving the file, i.e., it is a shortcut

    A1 = torch.io.Array('test1.hdf5')

    self.assertEqual ( A1.shape, (3, 3) )
    self.assertEqual ( A1.dtype, numpy.dtype('uint16') )
    self.assertTrue ( numpy.array_equal( A1.get()[1,:], [12, 19, 35] ) )

  def test03_set_and_get(self):

    # There is an interface for setting and getting the data within the Array
    # object. Here is how to use it.

    A = torch.io.Array([complex(1,1), complex(2,3.2), complex(3,5)], 'complex64')

    self.assertEqual ( A.shape, (3,) )
    self.assertEqual ( A.dtype, numpy.dtype('complex64') )

    # You can reset the Array contents with .set(). This totally resets the
    # internal contents of the Array.

    A.set(numpy.array([1,2,3], 'int8'))

    self.assertEqual ( A.dtype, numpy.dtype('int8') )
    self.assertTrue ( numpy.array_equal(A.get(), [1,2,3] ) )

  def test04_save_and_load(self):

    # You can save and load an Array from a file. 

    A1 = torch.io.Array('test1.hdf5')

    self.assertTrue  (A1.loadsAll)
    self.assertEqual (A1.filename, 'test1.hdf5')
    
    a_before = A1.get()
    
    A1.load()

    self.assertEqual (A1.filename, '')

    a_after = A1.get()

    self.assertTrue ( numpy.array_equal(a_after, a_before) )

    # Saving, off loads the contents on a file. Any new operation will read the
    # contents directly from the file.
    tname = tempname('.hdf5')

    A1.save(tname)

    self.assertTrue (A1.loadsAll)
    self.assertEqual (A1.filename, tname)

    self.assertTrue ( numpy.array_equal(A1.get(), a_before) )

    self.assertTrue (A1.filename, tname)

if __name__ == '__main__':
  import gc
  import inspect

  exclude = [
      "function",
      "type",
      "list",
      "dict",
      "tuple",
      "wrapper_descriptor",
      "module",
      "method_descriptor",
      "member_descriptor",
      "instancemethod",
      "builtin_function_or_method",
      "frame",
      "classmethod",
      "classmethod_descriptor",
      "_Environ",
      "MemoryError",
      "_Printer",
      "_Helper",
      "getset_descriptor",
      ]

  def dumpObjects():
    gc.collect()
    oo = gc.get_objects()
    for o in oo:
      if getattr(o, "__class__", None):
        name = o.__class__.__name__
        if name not in exclude:
          try:
            filename = inspect.getabsfile(o.__class__)
          except Exception, e:
            print "Cannot get filename of %s" % o.__class__.__name__
            continue

          print "Object of class:", name, "...",
          print "defined in file:", filename
  
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  os.chdir('data')

  suite = unittest.TestLoader().loadTestsFromTestCase(ArrayTest)
  unittest.TextTestRunner(verbosity=2).run(suite)
  
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

  #dumpObjects()
