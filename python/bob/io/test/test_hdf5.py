#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
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

"""Tests for the base HDF5 infrastructure
"""

import os, sys
import unittest
import numpy
import tempfile
import bob
import random
import pkg_resources

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def get_tempfilename(prefix='bobtest_', suffix='.hdf5'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def readWriteTest(self, outfile, dname, data, dtype=None):
  """Tests scalar input/output on HDF5 files"""

  # First, we test that we can read and write 1 single element
  outfile.append(dname + '_single', data[0], dtype=dtype)

  # Makes sure we can read the value out
  self.assertTrue( numpy.array_equal(outfile.lread(dname + '_single', 0), data[0]) )

  # Now we go for the full set
  outfile.append(dname, data, dtype=dtype)

  # And that we can read it back
  back = outfile.lread(dname) #we read all at once as it is simpler
  for i, b in enumerate(back): self.assertTrue( numpy.array_equal(b, data[i]) )

unittest.TestCase.readWriteTest = readWriteTest

def readWriteTestArray(self, outfile, dtype):
  N = 10
  SHAPE = (2, 3, 4, 2) #48 elements in arrays
  arrays = []
  for k in range(N):
    data = [random.uniform(0,N) for z in range(numpy.product(SHAPE))]
    nparray = numpy.array(data, dtype=dtype).reshape(SHAPE)
    arrays.append(nparray)
  self.readWriteTest(outfile, dtype + '_array', arrays)

unittest.TestCase.readWriteTestArray = readWriteTestArray

class HDF5FileTest(unittest.TestCase):
  """Performs various tests for the bob::io::HDF5File type."""
 
  def test01_CanCreate(self):
    # This test demonstrates how to create HDF5 files from scratch,
    # starting from blitz::Arrays

    # We start by creating some arrays to play with. Please note that in normal
    # cases you are either generating these arrays or reading from other binary
    # files or datasets.
    N = 2 
    SHAPE = (3, 2) #6 elements
    NELEMENT = SHAPE[0] * SHAPE[1]
    arrays = []
    for k in range(N):
      data = [int(random.uniform(0,10)) for z in range(NELEMENT)]
      arrays.append(numpy.array(data, 'int32').reshape(SHAPE))

    # Now we create a new binary output file in a temporary location and save
    # the data there.
    tmpname = get_tempfilename()
    outfile = bob.io.HDF5File(tmpname, 'w')
    outfile.append('testdata', arrays)

    # Data that is thrown in the file is immediately accessible, so you can
    # interleave read and write operations without any problems.
    # There is a single variable in the file, which is a bob arrayset:
    self.assertEqual(outfile.paths(), ['/testdata'])
    
    # And all the data is *exactly* the same recorded, bit by bit
    back = outfile.lread('testdata') # this is how to read the whole data back
    for i, b in enumerate(back):
      self.assertTrue( numpy.array_equal(b, arrays[i]) )

    # If you want to immediately close the HDF5 file, just delete the object
    del outfile

    # You can open the file in read-only mode using the 'r' flag. Writing
    # operations on this file will fail.
    readonly = bob.io.HDF5File(tmpname, 'r')

    # There is a single variable in the file, which is a bob arrayset:
    self.assertEqual(readonly.paths(), ['/testdata'])

    # You can get an overview of what is in the HDF5 dataset using the
    # describe() method
    description = readonly.describe('testdata')

    self.assertTrue(description[0].type.compatible(arrays[0]))
    self.assertEqual(description[0].size, N)

    # Test that writing will really fail
    self.assertRaises(RuntimeError, readonly.append, "testdata", arrays[0])

    # And all the data is *exactly* the same recorded, bit by bit
    back = readonly.lread('testdata') # this is how to read the whole data back
    for i, b in enumerate(back):
      self.assertTrue( numpy.array_equal(b, arrays[i]) )

    os.unlink(tmpname)

  def test02_TypeSupport(self):

    # This test will go through all supported types for reading/writing data
    # from to HDF5 files. One single file will hold all data for this test.
    # This is also supported with HDF5: multiple variables in a single file.

    N = 100 

    tmpname = get_tempfilename()
    outfile = bob.io.HDF5File(tmpname, 'w')

    data = [bool(int(random.uniform(0,2))) for z in range(N)]
    self.readWriteTest(outfile, 'bool_data', data)
    data = [int(random.uniform(0,100)) for z in range(N)]
    self.readWriteTest(outfile, 'int_data', data)
    self.readWriteTest(outfile, 'int8_data', data, 'int8')
    self.readWriteTest(outfile, 'uint8_data', data, 'uint8')
    self.readWriteTest(outfile, 'int16_data', data, 'int16')
    self.readWriteTest(outfile, 'uint16_data', data, 'uint16')
    self.readWriteTest(outfile, 'int32_data', data, 'int32')
    self.readWriteTest(outfile, 'uint32_data', data, 'uint32')

    data = [long(random.uniform(0,1000000000)) for z in range(N)]
    self.readWriteTest(outfile, 'long_data', data)
    self.readWriteTest(outfile, 'int64_data', data, 'int64')
    self.readWriteTest(outfile, 'uint64_data', data, 'uint64')

    data = [float(random.uniform(0,1)) for z in range(N)]
    self.readWriteTest(outfile, 'float_data', data)
    #Note that because of double => float precision issues, the next test will
    #fail. Python floats are actually double precision.
    #self.readWriteTest(outfile, 'float32_data', data, 'float32')
    self.readWriteTest(outfile, 'float64_data', data, 'float64')
    #The next construction is not supported by bob
    #self.readWriteTest(outfile, 'float128_data', data, 'float128')

    data = [complex(random.uniform(0,1),random.uniform(-1,0)) for z in range(N)]
    self.readWriteTest(outfile, 'complex_data', data)
    #Note that because of double => float precision issues, the next test will
    #fail. Python floats are actually double precision.
    #self.readWriteTest(outfile, 'complex64_data', data, 'complex64')
    self.readWriteTest(outfile, 'complex128_data', data, 'complex128')
    #The next construction is not supported by bob
    #self.readWriteTest(outfile, 'complex256_data', data, 'complex256')

    self.readWriteTestArray(outfile, 'int8')
    self.readWriteTestArray(outfile, 'int16')
    self.readWriteTestArray(outfile, 'int32')
    self.readWriteTestArray(outfile, 'int64')
    self.readWriteTestArray(outfile, 'uint8')
    self.readWriteTestArray(outfile, 'uint16')
    self.readWriteTestArray(outfile, 'uint32')
    self.readWriteTestArray(outfile, 'uint64')
    self.readWriteTestArray(outfile, 'float32')
    self.readWriteTestArray(outfile, 'float64')
    #self.readWriteTestArray(outfile, 'float128') #no numpy conversion
    self.readWriteTestArray(outfile, 'complex64')
    self.readWriteTestArray(outfile, 'complex128')
    #self.readWriteTestArray(outfile, 'complex256') #no numpy conversion

    os.unlink(tmpname)

  def test03_DatasetManagement(self):

    # This test examplifies dataset management within HDF5 files and how to
    # copy, delete and move data around.

    # Let's just create some dummy data to play with
    N = 100 

    tmpname = get_tempfilename()
    outfile = bob.io.HDF5File(tmpname, 'w')

    data = [int(random.uniform(0,N)) for z in range(N)]
    outfile.append('int_data', data)

    # This is how to rename a dataset.
    outfile.rename('int_data', 'MyRenamedDataset')
    
    # You can move the Dataset to any other hierarchy in the HDF5 file. The
    # directory structure within the file (i.e. the HDF5 groups) will be
    # created on demand.
    outfile.rename('MyRenamedDataset', 'NewDirectory1/Dir2/MyDataset')

    # Let's move the MyDataset dataset to another directory
    outfile.rename('NewDirectory1/Dir2/MyDataset', 'Test2/Bla')

    # So, now the original dataset name does not exist anymore
    self.assertEqual(outfile.paths(), ['/Test2/Bla'])

    # We can also unlink the dataset from the file. Please note this will not
    # erase the data in the file, just make it inaccessible
    outfile.unlink('Test2/Bla')
    
    # Finally, nothing is there anymore
    self.assertEqual(outfile.paths(), [])

    os.unlink(tmpname)

  def test04_resizeAndPreserve(self):
    
    # This test checks that non-contiguous C-style array can be saved
    # into an HDF5 file.

    # Let's just create some dummy data to play with
    SHAPE = (2, 3) #6 elements
    NELEMENT = SHAPE[0] * SHAPE[1]
    data = [int(random.uniform(0,10)) for z in range(NELEMENT)]
    array = numpy.array(data, 'int32').reshape(SHAPE)

    # Try to save a slice
    tmpname = get_tempfilename()
    bob.io.save(array[:,0], tmpname)

    os.unlink(tmpname)

  def test05_canLoadMatlab(self):

    # shows we can load a 2D matlab array and interpret it as a bunch of 1D
    # arrays, correctly

    t = bob.io.load(F('matlab_1d.hdf5'))
    self.assertEqual( t.shape, (512,) )
    self.assertEqual( t.dtype, numpy.float64 )

    t = bob.io.load(F('matlab_2d.hdf5'))
    self.assertEqual( t.shape, (512,2) )
    self.assertEqual( t.dtype, numpy.float64 )

    # interestingly enough, if you load those files as arrays, you will read
    # the whole data at once:

    t = bob.io.Array(F('matlab_1d.hdf5'))
    self.assertEqual( t.shape, (512,) )
    self.assertEqual( t.elementType.name, 'float64' )

    t = bob.io.Array(F('matlab_2d.hdf5'))
    self.assertEqual( t.shape, (512,2) )
    self.assertEqual( t.elementType.name, 'float64' )

  def test06_matlabImport(self):

    # This test verifies we can import HDF5 datasets generated in Matlab
    mfile = bob.io.HDF5File(F('matlab_1d.hdf5'))
    self.assertEqual ( mfile.paths(), ['/array'] )

  def test07_ioload_unlimited(self):

    # This test verifies that a 3D array whose first dimension is unlimited
    # and size equal to 1 can be read as a 2D array
    mfile = bob.io.load(F('test7_unlimited.hdf5'))
    self.assertTrue ( mfile.ndim == 2 )

  def test08_version(self):

    tmpname = get_tempfilename()
    outfile = bob.io.HDF5File(tmpname, 'w')
    self.assertEqual(outfile.version, None)
    outfile.version = 32
    self.assertEqual(outfile.version, 32)
    outfile.version = None
    self.assertEqual(outfile.version, None)

    os.unlink(tmpname)
