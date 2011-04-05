#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sun 20 Feb 07:49:14 2011 

"""A combined test for all built-in ArraysetCodecs in python.
"""

import os, sys
import unittest
import tempfile
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

  testcodec = torch.database.ArraysetCodecRegistry.getCodecByName(codecname)
  bincodec = torch.database.ArraysetCodecRegistry.getCodecByName("torch.arrayset.binary")

  # transcode to binary
  tmpname = tempname('.bin')
  bincodec.save(tmpname, testcodec.load(filename))
  self.assertEqual(bincodec.load(tmpname), testcodec.load(filename))

  # transcode to test format
  tmpname2 = tempname('.test')
  testcodec.save(tmpname2, bincodec.load(tmpname))
  self.assertEqual(testcodec.load(tmpname2), bincodec.load(tmpname))

  # And we erase both files after this
  os.unlink(tmpname)
  os.unlink(tmpname2)

# We attach the transcoding method to the testcase class, so we can use its
# assertions
unittest.TestCase.transcode = testcase_transcode

def testcase_readwrite(self, codecname, bzdata_list, save=None):
  """Runs a read/write verify step using the given bz data"""
  testcodec = torch.database.ArraysetCodecRegistry.getCodecByName(codecname)
  tmpname = tempname('.test')
  indata = torch.database.Arrayset()
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
  testcodec = torch.database.ArraysetCodecRegistry.getCodecByName(codecname)
  tmpname = tempname('.test')
  indata = torch.database.Arrayset()
  for k in bzdata_list:
    indata.append(k)
    testcodec.append(tmpname, torch.database.Array(k))

  #loads everything, see if are exactly the same
  reloaded = testcodec.load(tmpname)
  self.assertEqual(indata, reloaded)

  #loads one by one and checks they are individually correct
  for i, k in enumerate(bzdata_list):
    self.assertEqual(testcodec.load(tmpname, i+1).get(), i)

  os.unlink(tmpname)

# And we attach...
unittest.TestCase.append_load = testcase_append_load

# This is the data we test with
data_1 = [
    torch.core.array.float32_1(range(24), (24,)) / 24.,
    torch.core.array.float32_1(range(24), (24,)) / 48.,
    torch.core.array.float32_1(range(24), (24,)) / 0.25,
    ]
data_2 = [
    torch.core.array.float64_1(range(24), (24,)) / 24.3333334,
    torch.core.array.float64_1(range(24), (24,)) / -52.9,
    torch.core.array.float64_1(range(24), (24,)) / 37,
    ]
data_3 = [
    torch.core.array.complex128_3(range(24), (2,3,4)) / complex(24.3333334, 0.9),
    torch.core.array.complex128_3(range(24), (2,3,4)) / complex(0.1, -52.9),
    torch.core.array.complex128_3(range(24), (2,3,4)) / complex(37, -1e18),
    ]
data_4 = [
    torch.core.array.complex128_4(range(24000), (4,30,40,5)) / complex(24.3333334, 0.9),
    torch.core.array.complex128_4(range(24000), (4,30,40,5)) / complex(0.1, -52.9),
    torch.core.array.complex128_4(range(24000), (4,30,40,5)) / complex(37, -1e18),

    ]
data_4 = 100 * data_4 # 1'200 x 24'000 position complex<double> arrays

data_5 = [
    torch.core.array.complex128_4(range(240), (4,3,4,5)) / complex(24.3333334, 0.9),
    torch.core.array.complex128_4(range(240), (4,3,4,5)) / complex(0.1, -52.9),
    torch.core.array.complex128_4(range(240), (4,3,4,5)) / complex(37, -1e18),

    ]

class ArraysetCodecTest(unittest.TestCase):
  """Performs various tests for the Torch::database::*Codec* types."""

  def test01_binary(self):

    # The matlab codec accepts arbitrary input arrays if ints, floats, doubles
    # and complex values
    codec = "torch.arrayset.binary"
    self.readwrite(codec, data_1)
    self.readwrite(codec, data_2)
    self.readwrite(codec, data_3)
    self.readwrite(codec, data_4)
    self.append_load(codec, data_1)
    self.append_load(codec, data_2)
    self.append_load(codec, data_3)
    self.append_load(codec, data_4)
    self.transcode(codec, "test1.bin")

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
      testcodec = torch.database.ArraysetCodecRegistry.getCodecByName('matlab.arrayset.binary')
    except torch.database.CodecNotFound:
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
