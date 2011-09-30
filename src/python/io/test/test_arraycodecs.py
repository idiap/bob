#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sat 19 Feb 06:58:47 2011 

"""A combined test for all built-in ArrayCodecs in python.
"""

import os, sys
import unittest
import tempfile
import torch
import numpy
import random

# This test implements a generalized framework for testing Torch codecs. It
# loads files in the codec native format, convert into torch native binary
# format and back, comparing the outcomes at every stage. We believe in the
# quality of the binary codec because that is covered in other tests.

def tempname(suffix, prefix='torchtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def testcase_transcode(self, codecname, filename):
  """Runs a complete transcoding test, to and from the binary format."""

  testcodec = torch.io.ArrayCodecRegistry.getCodecByName(codecname)
  bincodec = torch.io.ArrayCodecRegistry.getCodecByName("hdf5.array.binary")

  # transcode to binary
  tmpname = tempname('.hdf5')
  bincodec.save(tmpname, testcodec.load(filename))
  self.assertTrue( numpy.array_equal(bincodec.load(tmpname).get(),
    testcodec.load(filename).get()) )

  # transcode to test format
  tmpname2 = tempname('.test')
  testcodec.save(tmpname2, bincodec.load(tmpname))
  self.assertTrue( numpy.array_equal(testcodec.load(tmpname2).get(),
    bincodec.load(tmpname).get()) )

  # And we erase both files after this
  os.unlink(tmpname)
  os.unlink(tmpname2)

# We attach the transcoding method to the testcase class, so we can use its
# assertions
unittest.TestCase.transcode = testcase_transcode

def testcase_readwrite(self, codecname, bzdata):
  """Runs a read/write verify step using the given bz data"""
  testcodec = torch.io.ArrayCodecRegistry.getCodecByName(codecname)
  tmpname = tempname('.test')
  testcodec.save(tmpname, torch.io.Array(bzdata))
  reloaded = testcodec.load(tmpname).get()
  self.assertTrue( numpy.array_equal(bzdata, reloaded) )
  os.unlink(tmpname)

def testcase_readwrite_ext(self, codecname, bzdata, ext):
  """Runs a read/write verify step using the given bz data"""
  testcodec = torch.io.ArrayCodecRegistry.getCodecByName(codecname)
  tmpname = tempname(ext)
  testcodec.save(tmpname, torch.io.Array(bzdata))
  reloaded = testcodec.load(tmpname).get()
  self.assertTrue( numpy.array_equal(bzdata, reloaded) )
  os.unlink(tmpname)

# And we attach...
unittest.TestCase.readwrite = testcase_readwrite
unittest.TestCase.readwrite_ext = testcase_readwrite_ext

class ArrayCodecTest(unittest.TestCase):
  """Performs various tests for the Torch::io::*Codec* types."""

  def test01_t3binary(self):
    self.readwrite("torch3.array.binary",
        numpy.array(range(24), dtype='float32').reshape(6,4) / 24.)
    self.readwrite("torch3.array.binary",
        numpy.array(range(24), dtype='float64').reshape(2,12) / 3.33336)
    self.transcode("torch3.array.binary", "torch3.bindata")

  def test02_matlab(self):
    try:
      testcodec = torch.io.ArrayCodecRegistry.getCodecByName('matlab.array.binary')
    except torch.io.CodecNotFound:
      #if the codec is not found, skip this test
      return
    self.readwrite("matlab.array.binary",
        numpy.array(range(24), dtype='float32').reshape(6,4) / 24.)
    self.readwrite("matlab.array.binary",
        numpy.array(range(24), dtype='float64').reshape(2,12) / 3.33336)
    self.readwrite("matlab.array.binary",
        numpy.array(range(24), dtype='complex64') / complex(29.5,37.2))
    self.readwrite("matlab.array.binary",
        numpy.array(range(24), dtype='complex128') / complex(12.7,-92))
    self.readwrite("matlab.array.binary",
        numpy.array(range(24), dtype='complex128').reshape(6,4) / complex(3.1416,-3.1416))
    self.transcode("matlab.array.binary", "test.mat")

  def test03_tensorfile(self):
    self.readwrite("torch.array.tensor",
        numpy.array(range(24), dtype='float32').reshape(6,4) / 24.)
    self.readwrite("torch.array.tensor",
        numpy.array(range(24), dtype='float64').reshape(2,12) / 3.33336)
    self.transcode("torch.array.tensor", "torch.tensor")

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
