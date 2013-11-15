#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Jul 22 07:59:06 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests bindings to the Visioner face localization framework.
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

  from .. import Localizer
  processor = Localizer()
  processor.detector.scanning_levels = 10
  locdata = processor(ip.rgb_to_gray(io.load(IMAGE)))
  assert locdata is not None

@utils.visioner_available
@utils.ffmpeg_found()
def test_faster():

  from .. import Localizer
  video = io.VideoReader(TEST_VIDEO)
  images = [ip.rgb_to_gray(k) for k in video[:20]]
  processor = Localizer()
  processor.detector.scanning_levels = 10

  # find faces on the video
  for image in images:
    locdata = processor(image)
    assert locdata is not None

@utils.visioner_available
@utils.ffmpeg_found()
def test_fast():

  from .. import Localizer
  video = io.VideoReader(TEST_VIDEO)
  images = [ip.rgb_to_gray(k) for k in video[:10]]
  processor = Localizer()
  processor.detector.scanning_levels = 5

  # find faces on the video
  for image in images:
    locdata = processor(image)
    assert locdata is not None

@utils.visioner_available
@utils.ffmpeg_found()
@nose.tools.nottest
def test_thorough():

  from .. import Localizer
  video = io.VideoReader(TEST_VIDEO)
  images = [ip.rgb_to_gray(k) for k in video[:2]]
  processor = Localizer()

  # find faces on the video
  for image in images:
    locdata = processor(image)
    assert locdata is not None
