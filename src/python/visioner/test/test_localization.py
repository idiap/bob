#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 22 Jul 07:38:58 2011 

"""Tests bindings to the Visioner face localization framework.
"""

import os
import sys
import unittest
import bob

TEST_VIDEO = "../../io/test/data/test.mov"

class Loader:
  
  def __init__(self):
    pass

  def setUp(self):
    if not hasattr(self, 'processor'):
      self.processor = bob.visioner.Localizer()

    if not hasattr(self, 'video'):
      self.video = bob.io.VideoReader(TEST_VIDEO)
      self.images = [bob.ip.rgb_to_gray(k).astype('int16') for k in self.video[:100]]

data = Loader()

class LocalizationTest(unittest.TestCase):
  """Performs various face localization tests."""

  def setUp(self):

    # load models and video only once
    if not hasattr(data, 'processor'): data.setUp()
    self.processor = data.processor
    self.images = data.images

  def test01_Thourough(self):

    # find faces on the video
    locdata = [self.processor(k) for k in self.images]

    # asserts at least 97% detections
    self.assertTrue ( (0.97 * len(self.images)) <= len(locdata) )

  def test02_Fast(self):

    # find faces on the video
    # scan_levels = 5, 8 scales
    self.processor.scan_levels = 5
    locdata = [self.processor(k) for k in self.images]

    # asserts at least 90% detections
    self.assertTrue ( (0.9 * len(self.images)) <= len(locdata) )

  def test03_Faster(self):

    # find faces on the video
    # scan_levels = 10, 8 scales
    self.processor.scan_levels = 10
    locdata = [self.processor(k) for k in self.images]

    # asserts at least 80% detections
    self.assertTrue ( (0.8 * len(self.images)) <= len(locdata) )

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()

