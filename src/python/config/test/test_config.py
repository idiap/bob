#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 30 Mar 13:48:19 2011 

"""Tests basic Configuration functionality available for the python bindings
"""

import os, sys, tempfile, random
import unittest
import torch
import numpy

DATADIR = os.path.join('..', '..', '..', 'cxx', 'config', 'test', 'data')
EX1 = os.path.join(DATADIR, 'example1.py')
EX2BROKEN = os.path.join(DATADIR, 'example2.py')

def tempname(suffix, prefix='torchtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def gen_array(dtype):
  SHAPE = (2, 3, 4, 2) #48 elements in arrays
  data = [random.uniform(0, 10) for z in range(numpy.product(SHAPE))]
  nparray = numpy.array(data).reshape(SHAPE)
  return torch.database.Array(torch.core.array.array(nparray, dtype))

def gen_arrayset(dtype):
  N = 10
  SHAPE = (2, 3, 4, 2) #48 elements in arrays
  aset = torch.database.Arrayset()
  for k in range(N):
    data = [random.uniform(0,N) for z in range(numpy.product(SHAPE))]
    nparray = numpy.array(data).reshape(SHAPE)
    aset.append(torch.core.array.array(nparray, dtype))
  return aset

def gen_vector(dtype):
  cls = getattr(torch.core.vector, dtype)
  obj = cls()
  obj[:] = [1, 2, 3, 4, 5, 6, 10, 9, 8, 7]
  return obj

def test_readwrite(self, var):
  """Tests a full chain write/read for HDF5 configuration support"""

  cfg = torch.config.Configuration()
  cfg['variable'] = var
  tmp = tempname('.hdf5')
  cfg.save(tmp)
  cfg2 = torch.config.Configuration(tmp) #re-open
  self.assertEqual(cfg2['variable'], var)
  os.unlink(tmp)
unittest.TestCase.hdf5ReadWrite = test_readwrite

def test_readwrite_single(self, var):
  """Tests a full chain write/read for HDF5 configuration support"""

  cfg = torch.config.Configuration()
  cfg['variable'] = var
  tmp = tempname('.hdf5')
  cfg.save(tmp)
  cfg2 = torch.config.Configuration(tmp) #re-open
  self.assertEqual(cfg2['variable'][0], var)
  os.unlink(tmp)
unittest.TestCase.hdf5ReadWriteSingle = test_readwrite_single

def test_readwrite_array(self, var):
  """Tests a full chain write/read for HDF5 configuration support"""

  cfg = torch.config.Configuration()
  cfg['variable'] = var
  tmp = tempname('.hdf5')
  cfg.save(tmp)
  cfg2 = torch.config.Configuration(tmp) #re-open
  self.assertEqual(cfg2['variable'][1], var)
  os.unlink(tmp)
unittest.TestCase.hdf5ReadWriteArray = test_readwrite_array

class ConfigTest(unittest.TestCase):
  """Various configuration tests."""

  def test01_canRetrieveFromPython(self):

    # In this example we show how the user can input data from an external
    # Torch configuration file written in python.

    # To create a new Configuration object, just construct one using as input
    # the filename where the configuration is written.
    c = torch.config.Configuration(EX1)

    # A Configuration object behaves like a python dictionary, so you can list,
    # iterate and display variables as you please

    # For example, you can use the 'in' operator to find if a certain variable
    # given a name is set. In our example configuration we have param1, 2, 3,
    # 4, and 5 there, so we check those were captured in correctly.
    self.assertTrue('param1' in c)
    self.assertTrue('param2' in c)
    self.assertTrue('param3' in c)
    self.assertTrue('param4' in c)
    self.assertTrue('param5' in c)

    # In C++ you need to use the "get()" method to retrieve the values, but in
    # Python, you access those using the [] dictionary operator.

    # This makes sure that param1 is the string I set in the file:
    self.assertEqual(c['param1'], "my test string")

    # And that param2 is a floating point number (close to pi)
    self.assertTrue(abs(c['param2']-3.1416) < 1e-5)

    # You can declare any objects in the python configuration file and these
    # objects can be of any type, as long as they are supported by the Torch
    # bindings, boost or any other (boost::python) library loaded by means of
    # import from inside of your configuration file.

    # Here we should you that 'param3' is actually a blitz::Array<int16_t,3>
    self.assertTrue(torch.core.array.is_blitz_array(c['param3']))
    self.assertEqual(c['param3'].extent(0), 2)
    self.assertEqual(c['param3'].extent(1), 3)
    self.assertEqual(c['param3'].extent(2), 4)

  def test02_canBuildInPython(self):

    # This test demonstrates the reverse way of the first test: It shows how
    # you build-up a new Configuration object from scratch. To start the
    # Configuration object, just build one w/o parameters:
    c = torch.config.Configuration()
    self.assertEqual(len(c), 0)

    # After you have created a Configuration object, you can add elements to it
    # by simply assigning them:
    c['value1'] = 2.78
    c['value2'] = torch.core.array.complex64_3(range(24), (2,3,4))
    self.assertEqual(len(c), 2)

    # You can also copy a whole configuration object within another by using
    # the "update()" method. This mechanism is also good for transcoding
    # configuration variables between formats.
    c2 = torch.config.Configuration(EX1)
    c.update(c2)
    self.assertEqual(len(c), 2 + len(c2))

    # You can reset values by just re-assigning
    self.assertEqual(type(c['value1']), float)
    c['value1'] = "another string"
    self.assertEqual(type(c['value1']), str)

    # TODO: Saving will not work presently
    # You can save (or as some prefer "serialize") the configuration contents
    # in an external file by using the "save()" command. The extension on the
    # file name will determine which particular storage implementation will be
    # used. We currently support only saving in HDF5 format.
    # c.save("test.hdf5")

  def test03_canModifyConfiguration(self):

    # As in a python dictionary, you can also modify the Configuration object
    # by reseting the elements or deleting them with the python standard del
    # command. Example:
    c = torch.config.Configuration(EX1)
    oldlength = len(c)
    del c['param1']
    self.assertEqual(len(c), oldlength - 1)
    self.assertTrue('param1' not in c)

    # You can clear all contents of a certain Configuration object by issuing
    # the "clear()" command:
    c.clear()
    self.assertEqual(len(c), 0)

  def test04_canDetectPythonSyntaxErrors(self):

    # Syntax errors on the configuration (python version) will be automatically
    # caught by the Python interpreter itself
    self.assertRaises(torch.config.PythonError, torch.config.Configuration, EX2BROKEN)

  def test05_canReadWriteHDF5Config(self):

    # This test examplifies how to read/write HDF5-based configuration files in
    # torch.
    config = torch.config.Configuration()
    
    # Test writing of all supported vector types
    #self.hdf5ReadWrite(gen_vector('bool')) #not supported by HDF5
    self.hdf5ReadWrite(gen_vector('int8'))
    self.hdf5ReadWrite(gen_vector('int16'))
    self.hdf5ReadWrite(gen_vector('int32'))
    self.hdf5ReadWrite(gen_vector('int64'))
    self.hdf5ReadWrite(gen_vector('uint8'))
    self.hdf5ReadWrite(gen_vector('uint16'))
    self.hdf5ReadWrite(gen_vector('uint32'))
    self.hdf5ReadWrite(gen_vector('uint64'))
    self.hdf5ReadWrite(gen_vector('float32'))
    self.hdf5ReadWrite(gen_vector('float64'))
    #self.hdf5ReadWrite(gen_vector('float128')) #not supported by Torch
    self.hdf5ReadWrite(gen_vector('complex64'))
    self.hdf5ReadWrite(gen_vector('complex128'))
    #self.hdf5ReadWrite(gen_vector('complex256')) #not supported by Torch

    # Test writing of all supported array/set types
    #self.hdf5ReadWrite(gen_arrayset('bool')) #not supported by HDF5
    self.hdf5ReadWrite(gen_arrayset('int8'))
    self.hdf5ReadWrite(gen_arrayset('int16'))
    self.hdf5ReadWrite(gen_arrayset('int32'))
    self.hdf5ReadWrite(gen_arrayset('int64'))
    self.hdf5ReadWrite(gen_arrayset('uint8'))
    self.hdf5ReadWrite(gen_arrayset('uint16'))
    self.hdf5ReadWrite(gen_arrayset('uint32'))
    self.hdf5ReadWrite(gen_arrayset('uint64'))
    self.hdf5ReadWrite(gen_arrayset('float32'))
    self.hdf5ReadWrite(gen_arrayset('float64'))
    # The following will not work in python because numpy does not support it
    #self.hdf5ReadWrite(gen_arrayset('float128'))
    self.hdf5ReadWrite(gen_arrayset('complex64'))
    self.hdf5ReadWrite(gen_arrayset('complex128'))
    # The following will not work in python because numpy does not support it
    #self.hdf5ReadWrite(gen_arrayset('complex256'))

    # This is a special case of vectors (single scalars)
    self.hdf5ReadWriteSingle(7) #integer
    self.hdf5ReadWriteSingle(3.14) #double
    self.hdf5ReadWriteSingle(complex(2, -1)) #complex<double>

    # This is a special case for arraysets (single arrays)
    #self.hdf5ReadWriteArray(gen_array('bool')) #not supported by HDF5
    self.hdf5ReadWriteArray(gen_array('int8'))
    self.hdf5ReadWriteArray(gen_array('int16'))
    self.hdf5ReadWriteArray(gen_array('int32'))
    self.hdf5ReadWriteArray(gen_array('int64'))
    self.hdf5ReadWriteArray(gen_array('uint8'))
    self.hdf5ReadWriteArray(gen_array('uint16'))
    self.hdf5ReadWriteArray(gen_array('uint32'))
    self.hdf5ReadWriteArray(gen_array('uint64'))
    self.hdf5ReadWriteArray(gen_array('float32'))
    self.hdf5ReadWriteArray(gen_array('float64'))
    # The following will not work in python because numpy does not support it
    #self.hdf5ReadWriteArray(gen_array('float128'))
    self.hdf5ReadWriteArray(gen_array('complex64'))
    self.hdf5ReadWriteArray(gen_array('complex128'))
    # The following will not work in python because numpy does not support it
    #self.hdf5ReadWriteArray(gen_array('complex256'))
    cname = tempname('.hdf5')
    config.save(cname)

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

