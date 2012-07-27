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

class DetectionTest(unittest.TestCase):
  """Performs various face detection tests."""
  
  def test01_Faster(self):

    video = bob.io.VideoReader(TEST_VIDEO)
    self.images = [bob.ip.rgb_to_gray(k) for k in video[:20]]
    self.processor = bob.visioner.MaxDetector(scanning_levels=10)

    # find faces on the video
    for image in self.images:
      locdata = self.processor(image)
      self.assertTrue(locdata is not None)

  def test02_Fast(self):

    video = bob.io.VideoReader(TEST_VIDEO)
    self.images = [bob.ip.rgb_to_gray(k) for k in video[:10]]
    self.processor = bob.visioner.MaxDetector(scanning_levels=5)

    # find faces on the video
    for image in self.images:
      locdata = self.processor(image)
      self.assertTrue(locdata is not None)

  def xtest03_Thorough(self):
    
    video = bob.io.VideoReader(TEST_VIDEO)
    self.images = [bob.ip.rgb_to_gray(k) for k in video[:2]]
    self.processor = bob.visioner.MaxDetector()

    # find faces on the video
    for image in self.images:
      locdata = self.processor(image)
      self.assertTrue(locdata is not None)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(DetectionTest)
