#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sat Nov 12 18:44:07 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Tests the io::Array interface from python.
"""

import os
import sys
import unittest
import bob
import numpy
import tempfile

def tempname(suffix, prefix='bobtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

class ArrayTest(unittest.TestCase):
  """Performs various tests for the bob::io::Array objects"""

  def test01_init_onthefly(self):

    # You can initialize io.Array's with NumPy ndarrays:
    a1 = numpy.array(range(4), 'float32').reshape(2,2)
    A1 = bob.io.Array(a1)

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
    A2 = bob.io.Array([[3, 5], [7, 9]])

    # That, of course, creates a new reference and a copy to the data, creating
    # internally its own numpy.ndarray and storing that.
    self.assertEqual ( A2.get().dtype, numpy.dtype('int') )

    # The form for creating an Array as above does not say anything about the
    # data type. We can do it by specifying it. Anything that is acceptable by
    # numpy.dtype() can be put in there:
    A3 = bob.io.Array([[3, 5], [7, 9]], 'float64')
    self.assertEqual ( A3.get().dtype, numpy.dtype('float64') )

  def test02_init_fromfiles(self):

    # Initialization from files is possible using the io.File object.
    f = bob.io.open('test1.hdf5', 'r')
    A1 = bob.io.Array(f) # reads all contents.

    self.assertEqual ( A1.shape, (3, 3) )
    self.assertEqual ( A1.dtype, numpy.dtype('uint16') )
    self.assertTrue ( numpy.array_equal( A1.get()[1,:], [12, 19, 35] ) )

    # You can also read files in "Arrayset" mode, in which we discretize the
    # first dimension. For example, reading a file that contains a single 2D
    # array [1, 2], [3, 4] at position 1 will actually only extract the 1D
    # array [3, 4]. See this:

    A1 = bob.io.Array(f, 1)
    
    self.assertEqual ( A1.shape, (3,) )
    self.assertEqual ( A1.dtype, numpy.dtype('uint16') )
    self.assertTrue ( numpy.array_equal( A1.get(), [12, 19, 35] ) )

    # Initializing by giving a filename is also possible, it is the same as
    # just giving the file, i.e., it is a shortcut

    A1 = bob.io.Array('test1.hdf5')

    self.assertEqual ( A1.shape, (3, 3) )
    self.assertEqual ( A1.dtype, numpy.dtype('uint16') )
    self.assertTrue ( numpy.array_equal( A1.get()[1,:], [12, 19, 35] ) )

  def test03_set_and_get(self):

    # There is an interface for setting and getting the data within the Array
    # object. Here is how to use it.

    A = bob.io.Array([complex(1,1), complex(2,3.2), complex(3,5)], 'complex64')

    self.assertEqual ( A.shape, (3,) )
    self.assertEqual ( A.dtype, numpy.dtype('complex64') )

    # You can reset the Array contents with .set(). This totally resets the
    # internal contents of the Array.

    A.set(numpy.array([1,2,3], 'int8'))

    self.assertEqual ( A.dtype, numpy.dtype('int8') )
    self.assertTrue ( numpy.array_equal(A.get(), [1,2,3] ) )

  def test04_save_and_load(self):

    # You can save and load an Array from a file. 

    A1 = bob.io.Array('test1.hdf5')

    self.assertTrue  (A1.loads_all)
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

    self.assertTrue (A1.loads_all)
    self.assertEqual (A1.filename, tname)

    self.assertTrue ( numpy.array_equal(A1.get(), a_before) )

    self.assertTrue (A1.filename, tname)

    os.unlink(tname)

  def test05_save_and_load_2Darrays(self):
    """Test introduced after ticket #105"""

    # Save and load a 2D arrays
    a=numpy.ndarray((3,1), 'float64')
    a[:,0]=[1,2,3]
    # Saving, off loads the contents on a file. Any new operation will read the
    # contents directly from the file.
    tname = tempname('.hdf5')
    bob.io.save(a, tname)
    b = bob.io.load(tname)
    self.assertTrue( numpy.array_equal(a, b) )
    os.unlink(tname)

    a=numpy.ndarray((1,3), 'float64')
    a[0,:]=[1,2,3]
    # Saving, off loads the contents on a file. Any new operation will read the
    # contents directly from the file.
    tname = tempname('.hdf5')
    bob.io.save(a, tname)
    b = bob.io.load(tname)
    self.assertTrue( numpy.array_equal(a, b) )
    os.unlink(tname)


def main():
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
 
  bob.helper.unittest_main(ArrayTest)()
  
  #dumpObjects()
