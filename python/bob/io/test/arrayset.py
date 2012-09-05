#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Aug 3 14:49:37 2011 +0200
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

"""Some tests to arraysets
"""

import os, sys
import unittest
import bob
import numpy
import tempfile

def tempname(suffix, prefix='bobtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

class ArraysetTest(unittest.TestCase):
  """Performs various tests for the bob::io::Arrayset objects"""

  def test01_initialization_with_iterables(self):

    # Another way to initialize Arraysets is with iterables:
    arrays = [[1,2,3], [4,5,6], [7.,3.14,19]]

    A = bob.io.Arrayset(arrays)

    self.assertEqual ( len(A), 3 )
    self.assertEqual ( A.dtype, numpy.dtype('int') ) # look at first array!

    # If you don't like that result, you can force the description type:

    B = bob.io.Arrayset(arrays, 'float')

    # On next test, note we can compare dtype's with a string!
    self.assertEqual ( B.dtype, 'float' )

    for k, array in enumerate(arrays):
      self.assertTrue ( numpy.array_equal(B[k], array) )

  def test02_initialization_with_files(self):

    # There are 2 ways to initialize an Arrayset with a file. Giving a range or
    # just letting it consume the whole file in one shot.

    A = bob.io.Arrayset('test1.hdf5', 'r') # use whole file.

    self.assertEqual ( len(A), 3 )
    self.assertEqual ( A.dtype, 'uint16' )

    # You can also limit the range: start at #1 instead of #0

    B = bob.io.Arrayset(bob.io.open('test1.hdf5', 'r'), 1) 

    self.assertEqual ( len(B), 2 )

    # Or limit start and end
    C = bob.io.Arrayset(bob.io.open('test1.hdf5', 'r'), 1, 2) #load only #1

    self.assertEqual ( len(C), 1 )

    self.assertTrue ( numpy.array_equal(C[0], [12, 19, 35]) )

    # As a shortcut to initialize with all data, you can use the filename. When
    # used in this mode, the input file is opened in 'a' (append) mode. This
    # will work fine normally, but if you are using the same filename
    # repeatedly like in this function, make sure you don't open it read-only
    # just before or HDF5 will not like it.

    D = bob.io.Arrayset('test1.hdf5')

    self.assertEqual ( len(D), 3 )
    self.assertEqual ( D.dtype, 'uint16' )
    self.assertEqual ( A, D )

  def test03_default_initialization (self):

    # Default initialization is when you start the Arrayset with no contents.
    A = bob.io.Arrayset()

    self.assertEqual ( len(A), 0 )
    self.assertRaises ( IndexError, A.__getitem__, 0 )

    # Please note that looking up the 'dtype' should reveal None in this case:
    self.assertEqual ( A.dtype, None )

    # When you push the first element, things get better established:
    A.append([1,2,3], 'complex128')

    self.assertEqual ( len(A), 1 )
    self.assertEqual ( A.dtype, 'complex128' )

  def test04_load_and_save (self):

    # Loading and saving works pretty much like for arrays. .load() loads all
    # in-file contents (if any) into memory while .save() does the inverse.
    
    A = bob.io.Arrayset('test1.hdf5', 'r') # use whole file.

    # Nothing should be loaded as of this time
    for k in range(len(A)):
      self.assertEqual ( A.get(k).filename, 'test1.hdf5' )
      self.assertEqual ( A.get(k).index, k )

    # Now we could load the data into memory:
    A.load()

    # What that did was just to invoke .load() on every internal io.Array.
    for k in range(len(A)):
      self.assertEqual ( A.get(k).filename, '' )

    # And we can save back into a temporary file.
    tname = tempname('.hdf5')

    A.save(tname)

    for k in range(len(A)):
      self.assertEqual ( A.get(k).filename, tname )
      self.assertEqual ( A.get(k).index, k )

    del A

    os.unlink(tname)

  def test05_list_operations (self):

    # Manipulations on Arraysets can either get you back io.Arrays or NumPy
    # ndarray's directly. Here is a demo and a test:

    A = bob.io.Arrayset('test1.hdf5', 'r') # use whole file.

    # .get(<id>) will return you always an io.Array.
    self.assertTrue ( isinstance(A.get(0), bob.io.Array) )

    # the [] operator will return you a numpy ndarray
    self.assertTrue ( isinstance(A[0], numpy.ndarray) )

    # Setting is a little bit easier. We internally switch on the context.
    A[1] = [4,5,6]

    A[0] = A.get(1) #setting with io.Array

    self.assertTrue ( numpy.array_equal(A[0], A[1]) )

    # You can also append:
    A.append([12, 24, 36])

    self.assertEqual ( len(A), 4 )
    self.assertTrue ( numpy.array_equal(A[3], [12, 24, 36]) )

    # Or delete a position.
    del A[0]

    self.assertEqual ( len(A), 3 )
    self.assertTrue ( numpy.array_equal(A[2], [12, 24, 36]) )

  def test10_extend1d(self):

    # shows how to use the extend() method on arraysets.

    t = bob.io.Arrayset()
    data = numpy.array(range(50), 'float32').reshape(25,2)
    t.extend(data, 0)

    self.assertEqual( len(t), 25 )
    self.assertEqual( t.elementType, bob.io.ElementType.float32 )
    self.assertEqual( t.shape, (2,) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(data[i,:], k) )

    # we can achieve the same effect with lists
    t = bob.io.Arrayset()
    vdata = [data[k,:] for k in range(data.shape[0])]
    t.extend(vdata)

    self.assertEqual( len(t), 25 )
    self.assertEqual( t.elementType, bob.io.ElementType.float32 )
    self.assertEqual( t.shape, (2,) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(vdata[i], k) )

  def test11_extend2d(self):

    # shows how to use the extend() method on arraysets.
    t = bob.io.Arrayset()
    data = numpy.array(range(90), 'float64').reshape(3,10,3)
    t.extend(data, 1)

    self.assertEqual( len(t), 10 )
    self.assertEqual( t.elementType, bob.io.ElementType.float64 )
    self.assertEqual( t.shape, (3,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(data[:,i,:], k) )

    # we can achieve the same effect with lists once more
    t = bob.io.Arrayset()
    vdata = [data[:,k,:] for k in range(data.shape[1])]
    t.extend(vdata)

    self.assertEqual( len(t), 10 )
    self.assertEqual( t.elementType, bob.io.ElementType.float64 )
    self.assertEqual( t.shape, (3,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(vdata[i], k) )

  def test12_extend3d(self):

    # shows how to use the extend() method on arraysets.

    t = bob.io.Arrayset()
    data = numpy.array(range(540), 'complex128').reshape(3,4,15,3)
    t.extend(data, 2)

    self.assertEqual( len(t), 15 )
    self.assertEqual( t.elementType, bob.io.ElementType.complex128 )
    self.assertEqual( t.shape, (3,4,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(data[:,:,i,:], k) )

    # we can achieve the same effect with lists once more
    t = bob.io.Arrayset()
    vdata = [data[:,:,k,:] for k in range(data.shape[2])]
    t.extend(vdata)

    self.assertEqual( len(t), 15 )
    self.assertEqual( t.elementType, bob.io.ElementType.complex128 )
    self.assertEqual( t.shape, (3,4,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(vdata[i], k) )

  def test13_openreadonly(self):

    # tests if the array can be opened in a read-only file system
    t = bob.io.Arrayset()
    data = numpy.array(range(540), 'complex128').reshape(3,4,15,3)
    t.extend(data, 2)
    name = tempname('.hdf5')
    t.save(name)

    # make this file read-only
    os.chmod(name, 0444)

    # now try to open it with read-write, see it fails
    self.assertRaises(RuntimeError, bob.io.Arrayset, name, 'a')

    # and we make sure it opens the file in read-only mode
    t2 = bob.io.Arrayset(name, 'r')
    self.assertEqual( len(t2), 15 )
    self.assertEqual( t2.elementType, bob.io.ElementType.complex128 )
    self.assertEqual( t2.shape, (3,4,3) )
    del t, t2
    os.unlink(name)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(ArraysetTest)
