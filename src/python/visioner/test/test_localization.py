#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 22 Jul 07:38:58 2011 

"""Tests bindings to the Visioner face localization framework.
"""

import os
import sys
import time
import unittest
import torch

OBJ_CLASSIF_MODEL = "data/Face.MCT9"
KEYP_LOC_MODEL = "data/Facial.MCT9.TMaxBoost"
TEST_VIDEO = "../../io/test/data/test.mov"

class LocalizationTest(unittest.TestCase):
  """Performs various face localization tests."""
  
  def test01_Thourough(self):

    # scan_levels = 0, 8 scales
    loc = torch.visioner.Localizer(OBJ_CLASSIF_MODEL, KEYP_LOC_MODEL)
    iv = torch.core.array.load(TEST_VIDEO) #4D uint8 array

    # do a gray-scale conversion for all frames
    images = [torch.ip.rgb_to_gray(k) for k in iv]

    # find faces on the video, time it
    start = time.clock()
    locdata = [loc(k) for k in images]
    total = time.clock() - start

    print "Located %d faces in %d frames" % (len(locdata), len(images))
    print "Total processing time is %.2e seconds" % total
    print "Estimated time per image is %.2e seconds" % (total/len(images))

  def test02_Fast(self):

    # scan_levels = 3, 8 scales
    loc = torch.visioner.Localizer(OBJ_CLASSIF_MODEL, KEYP_LOC_MODEL,
        scan_levels=3)
    iv = torch.core.array.load(TEST_VIDEO) #4D uint8 array

    # do a gray-scale conversion for all frames
    images = [torch.ip.rgb_to_gray(k) for k in iv]

    # find faces on the video, time it
    start = time.clock()
    locdata = [loc(k) for k in images]
    total = time.clock() - start

    print "Located %d faces in %d frames" % (len(locdata), len(images))
    print "Total processing time is %.2e seconds" % total
    print "Estimated time per image is %.2e seconds" % (total/len(images))

  def test02_Faster(self):

    # scan_levels = 3, 4 scales
    loc = torch.visioner.Localizer(OBJ_CLASSIF_MODEL, KEYP_LOC_MODEL,
        scan_levels=3, scale_var=4)
    iv = torch.core.array.load(TEST_VIDEO) #4D uint8 array

    # do a gray-scale conversion for all frames
    images = [torch.ip.rgb_to_gray(k) for k in iv]

    # find faces on the video, time it
    start = time.clock()
    locdata = [loc(k) for k in images]
    total = time.clock() - start

    print "Located %d faces in %d frames" % (len(locdata), len(images))
    print "Total processing time is %.2e seconds" % total
    print "Estimated time per image is %.2e seconds" % (total/len(images))

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

