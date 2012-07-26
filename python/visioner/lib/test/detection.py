#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sun Jul 24 16:57:47 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Tests bindings to the Visioner face detection framework.
"""

import os
import sys
import unittest
import bob

TEST_VIDEO = "test.mov"

class Loader:
  
  def __init__(self):
    pass

  def setUp(self):
    if not hasattr(self, 'processor'):
      self.processor = bob.visioner.MaxDetector()

    if not hasattr(self, 'video'):
      self.video = bob.io.VideoReader(TEST_VIDEO)
      self.images = [bob.ip.rgb_to_gray(k) for k in self.video[:100]]

data = Loader()

class DetectionTest(unittest.TestCase):
  """Performs various face detection tests."""
  
  def setUp(self):

    # load models and video only once
    if not hasattr(data, 'processor'): data.setUp()
    self.processor = data.processor
    self.images = data.images

  def test01_Thourough(self):

    # find faces on the video
    # scan_levels = 0, 8 scales
    locdata = [self.processor(k) for k in self.images]

    # asserts at least 95% detections
    self.assertTrue ( (0.95 * len(self.images)) <= len(locdata) )

  def test02_Fast(self):

    # find faces on the video
    # scan_levels = 3, 8 scales
    self.processor.scan_levels = 3
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

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(DetectionTest)
