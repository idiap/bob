#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 14 Apr 14:21:06 2011 

"""Tests for the base HDF5 infrastructure
"""

import os, sys
import unittest
import numpy
import tempfile
import torch
import random

def get_tempfilename(prefix='torchtest_', suffix='.hdf5'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def readWriteTest(self, outfile, dname, data, dtype=None):
  """Tests scalar input/output on HDF5 files"""

  # First, we test that we can read and write 1 single element
  outfile.append(dname + '_single', data[0], dtype=dtype)

  # Makes sure we can read the value out
  self.assertEqual(outfile.read(dname + '_single', 0), data[0])

  # Now we go for the full set
  outfile.append(dname, data, dtype=dtype)

  # And that we can read it back
  back = outfile.read(dname) #we read all at once as it is simpler
  for i, b in enumerate(back): self.assertEqual(b, data[i])

unittest.TestCase.readWriteTest = readWriteTest

def readWriteTestArray(self, outfile, dtype):
  N = 10
  SHAPE = (2, 3, 4, 2) #48 elements in arrays
  arrays = []
  for k in range(N):
    data = [random.uniform(0,N) for z in range(numpy.product(SHAPE))]
    nparray = numpy.array(data).reshape(SHAPE)
    arrays.append(torch.core.array.array(nparray, dtype))
  self.readWriteTest(outfile, dtype + '_array', arrays)

unittest.TestCase.readWriteTestArray = readWriteTestArray

class HDF5FileTest(unittest.TestCase):
  """Performs various tests for the Torch::database::HDF5File type."""
 
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
      arrays.append(torch.core.array.int32_2(data, SHAPE))

    # Now we create a new binary output file in a temporary location and save
    # the data there.
    tmpname = get_tempfilename()
    outfile = torch.database.HDF5File(tmpname)
    outfile.append('testdata', arrays)

    # Data that is thrown in the file is immediately accessible, so you can
    # interleave read and write operations without any problems.
    # There is a single variable in the file, which is a torch arrayset:
    self.assertEqual(outfile.paths(), ['/testdata'])
    
    # And all the data is *exactly* the same recorded, bit by bit
    back = outfile.read('testdata') # this is how to read the whole data back
    for i, b in enumerate(back):
      self.assertTrue( (b == arrays[i]).all() )

    # If you want to immediately close the HDF5 file, just delete the object
    del outfile

    # You can open the file in read-only mode using the 'r' flag. Writing
    # operations on this file will fail.
    readonly = torch.database.HDF5File(tmpname, 'r')

    # There is a single variable in the file, which is a torch arrayset:
    self.assertEqual(readonly.paths(), ['/testdata'])

    # You can get an overview of what is in the HDF5 dataset using the
    # describe() method
    description = readonly.describe('testdata')

    self.assertTrue(description.compatible(arrays[0]))
    self.assertEqual(readonly.size('testdata'), N)

    # Test that writing will really fail
    self.assertRaises(torch.database.HDF5StatusError, readonly.append,
        "testdata", arrays[0])

    # And all the data is *exactly* the same recorded, bit by bit
    back = readonly.read('testdata') # this is how to read the whole data back
    for i, b in enumerate(back):
      self.assertTrue( (b == arrays[i]).all() )

    os.unlink(tmpname)

  def test02_TypeSupport(self):

    # This test will go through all supported types for reading/writing data
    # from to HDF5 files. One single file will hold all data for this test.
    # This is also supported with HDF5: multiple variables in a single file.

    N = 100 

    tmpname = get_tempfilename()
    outfile = torch.database.HDF5File(tmpname)

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
    #The next construction is not supported by Torch
    #self.readWriteTest(outfile, 'float128_data', data, 'float128')

    data = [complex(random.uniform(0,1),random.uniform(-1,0)) for z in range(N)]
    self.readWriteTest(outfile, 'complex_data', data)
    #Note that because of double => float precision issues, the next test will
    #fail. Python floats are actually double precision.
    #self.readWriteTest(outfile, 'complex64_data', data, 'complex64')
    self.readWriteTest(outfile, 'complex128_data', data, 'complex128')
    #The next construction is not supported by Torch
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
    outfile = torch.database.HDF5File(tmpname)

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

  def test04_directory_support(self):

    """
    conf::Configuration c;
    int value = 10;

    c.set("value", value);
    check_equal(c, "value", value);
    
    c.set("a/value", value);
    check_equal(c, "a/value", value);
    
    c.cd("b");
    c.set("value", value);
    check_equal(c, "/b/value", value);

    c.set("c/value", value);
    check_equal(c, "/b/c/value", value);

    c.cd("d");
    c.set("value", value);
    check_equal(c, "/b/d/value", value);

    c.cd("../e");
    c.set("value", value);
    check_equal(c, "/b/e/value", value);

    c.set("/f/value", value);
    check_equal(c, "/f/value", value);
    c.cd("../..");
    check_equal(c, "f/value", value);

    c.cd("..");
    c.set("/g/value", value);
    check_equal(c, "/g/value", value);

    c.cd("b/d");
    c.cd("/h");
    c.set("value", value);
    check_equal(c, "/h/value", value);
    """

    pass

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  os.chdir('data')
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

