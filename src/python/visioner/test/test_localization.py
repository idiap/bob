#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 22 Jul 07:38:58 2011 

"""Tests bindings to the Visioner face localization framework.
"""

import os
import sys
import unittest
import torch

TEST_VIDEO = "../../io/test/data/test.mov"

class LocalizationTest(unittest.TestCase):
  """Performs various face localization tests."""
  
  def test00_one(self):

    # scan_levels = 0, 8 scales
    loc = torch.visioner.Localizer()
    iv = torch.io.VideoReader(TEST_VIDEO)

    # do a gray-scale conversion for all frames and cast to int16
    #print "Converting video..."
    images = [torch.ip.rgb_to_gray(k).cast('int16') for k in iv[:100]]

    # find faces on the video
    locdata = [loc(k) for k in images]

    # asserts at least 97% detections
    self.assertTrue ( (0.97 * len(images)) <= len(locdata) )

  def test01_Thourough(self):

    # scan_levels = 0, 8 scales
    loc = torch.visioner.Localizer()
    iv = torch.io.VideoReader(TEST_VIDEO)

    # do a gray-scale conversion for all frames and cast to int16
    images = [torch.ip.rgb_to_gray(k).cast('int16') for k in iv]

    # find faces on the video
    locdata = [loc(k) for k in images]

    # asserts at least 97% detections
    print len(locdata)
    self.assertTrue ( (0.97 * len(images)) <= len(locdata) )

  def xtest02_Fast(self):

    # TODO: temporarily disabled due to the slowness of model loading

    # scan_levels = 3, 8 scales
    loc = torch.visioner.Localizer(scan_levels=3)
    iv = torch.core.array.load(TEST_VIDEO) #4D uint8 array

    # do a gray-scale conversion for all frames
    images = [torch.ip.rgb_to_gray(k).cast('int16') for k in iv]

    # find faces on the video
    locdata = [loc(k) for k in images]

    # asserts at least 93% detections
    self.asserttrue ( (0.93 * len(images)) > len(locdata) )

  def xtest03_Faster(self):

    # TODO: temporarily disabled due to the slowness of model loading

    # scan_levels = 3, 4 scales
    loc = torch.visioner.Localizer(scan_levels=3, scale_var=4)
    iv = torch.core.array.load(TEST_VIDEO) #4D uint8 array

    # do a gray-scale conversion for all frames
    images = [torch.ip.rgb_to_gray(k).cast('int16') for k in iv]

    # find faces on the video
    locdata = [loc(k) for k in images]

    # asserts at least 90% detections
    self.assertTrue ( (0.9 * len(images)) > len(locdata) )

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

