#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 13 Jul 2010 09:36:54 CEST 

"""Runs some video tests
"""

# These are some global parameters for the test.
INPUT_VIDEO = 'test.mov'
OUTPUT_VIDEO = '/tmp/video_test.mov'

import unittest
import torch

class VideoTest(unittest.TestCase):
  """Performs various combined read/write tests on the Torch::Video object"""
  
  def test01_CanOpen(self):
    v = torch.ip.Video(INPUT_VIDEO) 
    self.assertEqual(v.getState(), torch.ip.State.Read)

  def test02_CanReadGrayImage(self):
    v = torch.ip.Video(INPUT_VIDEO) 
    self.assertEqual(v.getState(), torch.ip.State.Read)
    i = torch.ip.Image(1, 1, 1)
    self.assertEqual(v.read(i), True)
    self.assertEqual(i.getWidth(), v.width())
    self.assertEqual(i.getHeight(), v.height())
    self.assertEqual(i.getNPlanes(), 1)

  def test03_CanReadMultiColorImage(self):
    v = torch.ip.Video(INPUT_VIDEO) 
    i = torch.ip.Image(1, 1, 3)
    self.assertEqual(v.read(i), True)
    self.assertEqual(i.getWidth(), v.width())
    self.assertEqual(i.getHeight(), v.height())
    self.assertEqual(i.getNPlanes(), 3)
    self.assertEqual(v.getState(), torch.ip.State.Read)

  def test04_CanReadManyImages(self):
    v = torch.ip.Video(INPUT_VIDEO) 
    i = torch.ip.Image(1, 1, 1)
    for k in range(10):
      self.assertEqual(v.read(i), True)
      self.assertEqual(i.getWidth(), v.width())
      self.assertEqual(i.getHeight(), v.height())
      self.assertEqual(i.getNPlanes(), 1)
    self.assertEqual(v.getState(), torch.ip.State.Read)

  def test05_CanWriteVideo(self):
    iv = torch.ip.Video(INPUT_VIDEO)
    ov = torch.ip.Video(OUTPUT_VIDEO, iv) #makes a video like the other 
    i = torch.ip.Image(1, 1, 1)
    self.assertEqual(iv.read(i), True)
    for k in range(50): self.assertEqual(ov.write(i), True)
    self.assertEqual(iv.getState(), torch.ip.State.Read)
    self.assertEqual(ov.getState(), torch.ip.State.Write)

if __name__ == '__main__':
  import os, sys
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
