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
  tmpname = tempname('.bindata')
  testcodec.save(tmpname, torch.database.Array(bzdata))
  reloaded = testcodec.load(tmpname).get()
  self.assertEqual(bzdata, reloaded)
unittest.TestCase.readwrite = testcase_readwrite

class ArrayCodecTest(unittest.TestCase):
  """Performs various tests for the Torch::database::*Codec* types."""

  def test01_t3binary(self):
    self.readwrite("torch3.array.binary",
        torch.core.array.float32_1(range(24), (24,)) / 24.)
    self.readwrite("torch3.array.binary",
        torch.core.array.float64_1(range(24), (24,)) / 3.33336)
    self.transcode("torch3.array.binary", "torch3.bindata")

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
