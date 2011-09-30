#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sun 20 Feb 07:49:14 2011 

"""A combined test for all built-in ArraysetCodecs in python.
"""

import os, sys
import unittest
import tempfile
import numpy
import torch
import random

# This test implements a generalized framework for testing Torch codecs. It
# loads files in the codec native format, convert into torch native binary
# format and back, comparing the outcomes at every stage. 

def tempname(suffix, prefix='torchtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def testcase_transcode(self, codecname, filename):
  """Runs a complete transcoding test, to and from the binary format."""

  testcodec = torch.io.ArraysetCodecRegistry.getCodecByName(codecname)
  bincodec = torch.io.ArraysetCodecRegistry.getCodecByName("hdf5.arrayset.binary")

  # transcode to binary
  tmpname = tempname('.hdf5')
  bincodec.save(tmpname, testcodec.load(filename))
  self.assertTrue( numpy.array_equal(bincodec.load(tmpname), testcodec.load(filename)) )

  # transcode to test format
  tmpname2 = tempname('.test')
  testcodec.save(tmpname2, bincodec.load(tmpname))
  self.assertTrue( numpy.array_equal(testcodec.load(tmpname2), bincodec.load(tmpname)) )

  # And we erase both files after this
  os.unlink(tmpname)
  os.unlink(tmpname2)

# We attach the transcoding method to the testcase class, so we can use its
# assertions
unittest.TestCase.transcode = testcase_transcode

def testcase_readwrite(self, codecname, bzdata_list, save=None):
  """Runs a read/write verify step using the given bz data"""
  testcodec = torch.io.ArraysetCodecRegistry.getCodecByName(codecname)
  tmpname = tempname('.test')
  indata = torch.io.Arrayset()
  for k in bzdata_list: indata.append(k)
  testcodec.save(tmpname, indata)

  # just peeking
  (eltype, shape, samples) = testcodec.peek(tmpname)
  self.assertEqual(eltype, indata.elementType)
  self.assertEqual(shape, indata.shape)
  self.assertEqual(samples, len(indata))

  # full loading
  reloaded = testcodec.load(tmpname)
  self.assertEqual(indata, reloaded)
  if save and isinstance(save, (str, unicode)):
    import shutil
    shutil.copy(tmpname, save)
  os.unlink(tmpname)

# And we attach...
unittest.TestCase.readwrite = testcase_readwrite

def testcase_append_load(self, codecname, bzdata_list):
  """Runs an arrayset append() on the codec, tests if it keeps the data."""
  testcodec = torch.io.ArraysetCodecRegistry.getCodecByName(codecname)
  tmpname = tempname('.test')
  indata = torch.io.Arrayset()
  for k in bzdata_list:
    indata.append(k)
    testcodec.append(tmpname, torch.io.Array(k))

  #loads everything, see if are exactly the same
  reloaded = testcodec.load(tmpname)
  self.assertTrue( numpy.array_equal(indata, reloaded) )

  #loads one by one and checks they are individually correct
  for i, k in enumerate(bzdata_list):
    self.assertTrue( numpy.array_equal(testcodec.load(tmpname, i).get(), k) )

  os.unlink(tmpname)

# And we attach...
unittest.TestCase.append_load = testcase_append_load

# This is the data we test with
data_1 = [
    numpy.array(range(24), 'float32') / 24.,
    numpy.array(range(24), 'float32') / 48.,
    numpy.array(range(24), 'float32') / 0.25,
    ]
data_2 = [
    numpy.array(range(24), 'float64') / 24.33333334,
    numpy.array(range(24), 'float64') / -52.9,
    numpy.array(range(24), 'float64') / 37,
    ]
data_3 = [
    numpy.array(range(24), 'complex128').reshape(2,3,4) / complex(24.33333334, 0.9),
    numpy.array(range(24), 'complex128').reshape(2,3,4) / complex(0.1, -52.9),
    numpy.array(range(24), 'complex128').reshape(2,3,4) / complex(37, -1e18),
    ]
data_4 = [
    numpy.array(range(24000), 'complex128').reshape(4,30,40,5) / complex(24.3333334, 0.9),
    numpy.array(range(24000), 'complex128').reshape(4,30,40,5) / complex(0.1, -52.9),
    numpy.array(range(24000), 'complex128').reshape(4,30,40,5) / complex(37, -1e18),
    ]
data_4 = 100 * data_4 # 1'200 x 24'000 position complex<double> arrays

data_5 = [
    numpy.array(range(240), 'complex128').reshape(4,3,4,5) / complex(24.3333334, 0.9),
    numpy.array(range(240), 'complex128').reshape(4,3,4,5) / complex(0.1, -52.9),
    numpy.array(range(240), 'complex128').reshape(4,3,4,5) / complex(37, -1e18),
    ]

class ArraysetCodecTest(unittest.TestCase):
  """Performs various tests for the Torch::io::*Codec* types."""

  def test01_binary(self):

    # The matlab codec accepts arbitrary input arrays if ints, floats, doubles
    # and complex values
    codec = "hdf5.arrayset.binary"
    self.readwrite(codec, data_1)
    self.readwrite(codec, data_2)
    self.readwrite(codec, data_3)
    self.readwrite(codec, data_4)
    self.append_load(codec, data_1)
    self.append_load(codec, data_2)
    self.append_load(codec, data_3)
    self.append_load(codec, data_4)
    self.transcode(codec, "test1.hdf5")

  def test02_t3binary(self):

    # The torch3 file format only accepts single dimension floats or doubles
    codec = "torch3.arrayset.binary"
    self.readwrite(codec, data_1)
    self.readwrite(codec, data_2)
    self.append_load(codec, data_1)
    self.append_load(codec, data_2)
    self.transcode(codec, "torch3.bindata")

  def test03_matlab(self):
    
    try:
      testcodec = torch.io.ArraysetCodecRegistry.getCodecByName('matlab.arrayset.binary')
    except torch.io.CodecNotFound:
      #if the codec is not found, skip this test
      return

    # The matlab codec accepts arbitrary input arrays if ints, floats, doubles
    # and complex values
    codec = "matlab.arrayset.binary"
    self.readwrite(codec, data_1)
    self.readwrite(codec, data_2)
    self.readwrite(codec, data_3)
    self.readwrite(codec, data_4)
    self.append_load(codec, data_1)
    self.append_load(codec, data_2)
    self.append_load(codec, data_3)
    self.append_load(codec, data_4)
    self.transcode(codec, "test.mat")

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
