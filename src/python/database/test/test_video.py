#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 13 Jul 2010 09:36:54 CEST 

"""Runs some video tests
"""

import os, sys
import tempfile

def test_file(name):
  """Returns the path to the filename for this test."""
  return os.path.join("data", name)

def get_tempfilename(prefix='torchtest_', suffix='.avi'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.unlink(name)
  return name

# These are some global parameters for the test.
INPUT_VIDEO = test_file('test.mov')
OUTPUT_VIDEO = get_tempfilename()

import unittest
import torch

class VideoTest(unittest.TestCase):
  """Performs various combined read/write tests on video files"""
  
  def test01_CanOpen(self):

    # This test opens and verifies some properties of the test video available.
    # It examplifies how to directly call the VideoReader and how to access
    # some of its properties.
    v = torch.database.VideoReader(INPUT_VIDEO)
    self.assertEqual(v.height, 240)
    self.assertEqual(v.width, 320)
    self.assertEqual(v.duration, 15000000) #microseconds
    self.assertEqual(v.numberOfFrames, 375)
    self.assertEqual(len(v), 375)
    self.assertEqual(v.codecName, 'mjpeg')

  def test02_CanReadImages(self):

    # This test shows how you can read image frames from a VideoReader
    v = torch.database.VideoReader(INPUT_VIDEO)
    for frame in v:
      # Note that when you iterate, the frames are blitz::Array<> objects
      # So, you can use them as you please. The organization of the data
      # follows the other encoding systems in torch: (color-bands, height,
      # width).
      self.assertTrue(torch.core.array.is_blitz_array(frame))
      self.assertEqual(frame.dimensions(), 3)
      self.assertEqual(frame.extent(0), 3) #color-bands (RGB)
      self.assertEqual(frame.extent(1), 240) #height
      self.assertEqual(frame.extent(2), 320) #width

  def test03_CanWriteVideo(self):

    # This test reads all frames in sequence from a initial video and records
    # them into an output video, possibly transcoding it.
    iv = torch.database.VideoReader(INPUT_VIDEO)
    ov = torch.database.VideoWriter(OUTPUT_VIDEO, iv.height, iv.width)
    for frame in iv: ov.append(frame)
    
    # We verify that both videos have similar properties
    self.assertEqual(len(iv), len(ov))
    self.assertEqual(iv.width, ov.width)
    self.assertEqual(iv.height, ov.height)
    
    del ov # trigger closing of the output video stream

    iv2 = torch.database.VideoReader(OUTPUT_VIDEO)

    # We verify that both videos have similar frames
    for orig, copied in zip(iv.__iter__(), iv2.__iter__()):
      diff = abs(orig.cast('float32')-copied.cast('float32'))
      m = diff.mean()
      self.assertTrue(m < 3.0) # average difference is less than 3 gray levels
    os.unlink(OUTPUT_VIDEO)

if __name__ == '__main__':
  import sys
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
