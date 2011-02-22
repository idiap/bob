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

  testcodec = torch.database.ArrayCodecRegistry.getCodecByName(codecname)
  bincodec = torch.database.ArrayCodecRegistry.getCodecByName("torch.array.binary")

  # transcode to binary
  tmpname = tempname('.bin')
  bincodec.save(tmpname, testcodec.load(filename))
  self.assertEqual(bincodec.load(tmpname).get(), testcodec.load(filename).get())

  # transcode to test format
  tmpname2 = tempname('.test')
  testcodec.save(tmpname2, bincodec.load(tmpname))
  self.assertEqual(testcodec.load(tmpname2).get(), bincodec.load(tmpname).get())

  # And we erase both files after this
  os.unlink(tmpname)
  os.unlink(tmpname2)

# We attach the transcoding method to the testcase class, so we can use its
# assertions
unittest.TestCase.transcode = testcase_transcode

def testcase_readwrite(self, codecname, bzdata):
  """Runs a read/write verify step using the given bz data"""
  testcodec = torch.database.ArrayCodecRegistry.getCodecByName(codecname)
  tmpname = tempname('.test')
  testcodec.save(tmpname, torch.database.Array(bzdata))
  reloaded = testcodec.load(tmpname).get()
  self.assertEqual(bzdata, reloaded)
  os.unlink(tmpname)

# And we attach...
unittest.TestCase.readwrite = testcase_readwrite

class ArrayCodecTest(unittest.TestCase):
  """Performs various tests for the Torch::database::*Codec* types."""

  def test01_t3binary(self):
    self.readwrite("torch3.array.binary",
        torch.core.array.float32_1(range(24), (24,)) / 24.)
    self.readwrite("torch3.array.binary",
        torch.core.array.float64_1(range(24), (24,)) / 3.33336)
    self.transcode("torch3.array.binary", "torch3.bindata")

  def test02_matlab(self):
    self.readwrite("matlab.array.binary",
        torch.core.array.float32_1(range(24), (24,)) / 24.)
    self.readwrite("matlab.array.binary",
        torch.core.array.float64_1(range(24), (24,)) / 3.33336)
    self.readwrite("matlab.array.binary",
        torch.core.array.complex64_1(range(24), (24,)) / complex(29.5,37.2))
    self.readwrite("matlab.array.binary",
        torch.core.array.complex128_1(range(24), (24,)) / complex(12.7,-92))
    self.readwrite("matlab.array.binary",
        torch.core.array.complex128_2(range(24), (6,4)) / complex(3.1416,-3.1416))
    self.transcode("matlab.array.binary", "test.mat")

  def test03_imagepbm(self):
    self.readwrite("torch.image.pbm_p4",
      torch.core.array.uint8_3(range(24), (1,6,4)))
    self.transcode("torch.image.pbm_p4", "test.pbm")

  def test04_imagepgm(self):
    self.readwrite("torch.image.pgm_p5",
      torch.core.array.uint8_3(range(24), (1,6,4)) * 4)
    self.readwrite("torch.image.pgm_p5",
      torch.core.array.uint8_3(range(24), (1,6,4)) * 7)
    self.transcode("torch.image.pgm_p5", "test.pgm")

  def test05_imageppm(self):
    self.readwrite("torch.image.ppm_p6",
      torch.core.array.uint8_3(range(24), (3,2,4)) * 4)
    self.readwrite("torch.image.ppm_p6",
      torch.core.array.uint8_3(range(24), (3,2,4)) * 7)
    self.transcode("torch.image.ppm_p6", "test.ppm")

  def test06_imagejpg(self):
    self.readwrite("torch.image.jpg",
      torch.core.array.uint8_3(range(24), (3,2,4)) * 4)
    self.readwrite("torch.image.jpg",
      torch.core.array.uint8_3(range(24), (3,2,4)) * 7)
    self.transcode("torch.image.jpg", "test.jpg")


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
