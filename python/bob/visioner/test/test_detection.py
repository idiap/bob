#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sun Jul 24 16:57:47 2011 +0200
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

"""Tests bindings to the Visioner face detection framework.
"""

import os
import nose.tools
from ...test import utils
from ... import io, ip
from ...ip import test as iptest
from ...io import test as iotest

TEST_VIDEO = utils.datafile("test.mov", iotest.__name__)
IMAGE = utils.datafile('test-faces.jpg', iptest.__name__, os.path.join('data', 'faceextract'))

@utils.visioner_available
def test_single():

  from .. import MaxDetector
  processor = MaxDetector(scanning_levels=10)
  locdata = processor(ip.rgb_to_gray(io.load(IMAGE)))
  assert locdata is not None

@utils.visioner_available
@utils.ffmpeg_found()
def test_faster():

  from .. import MaxDetector
  video = io.VideoReader(TEST_VIDEO)
  images = [ip.rgb_to_gray(k) for k in video[:20]]
  processor = MaxDetector(scanning_levels=10)

  # find faces on the video
  for image in images:
    locdata = processor(image)
    assert locdata is not None

@utils.visioner_available
@utils.ffmpeg_found()
@nose.tools.nottest
def test_fast():

  from .. import MaxDetector
  video = io.VideoReader(TEST_VIDEO)
  images = [ip.rgb_to_gray(k) for k in video[:10]]
  processor = MaxDetector(scanning_levels=5)

  # find faces on the video
  for image in images:
    locdata = processor(image)
    assert locdata is not None

@utils.visioner_available
@utils.ffmpeg_found()
@nose.tools.nottest
def test_thourough():

  from .. import MaxDetector
  video = io.VideoReader(TEST_VIDEO)
  images = [ip.rgb_to_gray(k) for k in video[:2]]
  processor = MaxDetector()

  # find faces on the video
  for image in images:
    locdata = processor(image)
    assert locdata is not None
