#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Jul 22 07:59:06 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

"""Tests bindings to the Visioner face localization framework.
"""

import os
import unittest
from ...test import utils
from ... import io, ip
from ...ip import test as iptest
from ...io import test as iotest
from .. import Localizer

TEST_VIDEO = utils.datafile("test.mov", iotest)
IMAGE = utils.datafile('test-faces.jpg', iptest, os.path.join('data', 'faceextract'))

class LocalizationTest(unittest.TestCase):
  """Performs various face localization tests."""

  def test00_Single(self):

    self.processor = Localizer()
    self.processor.detector.scanning_levels = 10
    locdata = self.processor(ip.rgb_to_gray(io.load(IMAGE)))
    self.assertTrue(locdata is not None)

  @utils.ffmpeg_found()
  def test01_Faster(self):

    video = io.VideoReader(TEST_VIDEO)
    self.images = [ip.rgb_to_gray(k) for k in video[:20]]
    self.processor = Localizer()
    self.processor.detector.scanning_levels = 10

    # find faces on the video
    for image in self.images:
      locdata = self.processor(image)
      self.assertTrue(locdata is not None)

  @utils.ffmpeg_found()
  def test02_Fast(self):

    video = io.VideoReader(TEST_VIDEO)
    self.images = [ip.rgb_to_gray(k) for k in video[:10]]
    self.processor = Localizer()
    self.processor.detector.scanning_levels = 5

    # find faces on the video
    for image in self.images:
      locdata = self.processor(image)
      self.assertTrue(locdata is not None)

  @utils.ffmpeg_found()
  def xtest03_Thorough(self):
      
    video = io.VideoReader(TEST_VIDEO)
    self.images = [ip.rgb_to_gray(k) for k in video[:2]]
    self.processor = Localizer()

    # find faces on the video
    for image in self.images:
      locdata = self.processor(image)
      self.assertTrue(locdata is not None)
