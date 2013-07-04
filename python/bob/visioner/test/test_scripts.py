#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:30:54 CEST

"""Test scripts for the visioner
"""

import os
from ...test import utils
from ..script import facebox, facepoints
from ...ip import test as iptest
from ...io import test as iotest

MOVIE = utils.datafile('test.mov', iotest.__name__)
IMAGE = utils.datafile('test-faces.jpg', iptest.__name__, os.path.join('data', 'faceextract'))

@utils.visioner_available
@utils.ffmpeg_found()
def test_face_detect_on_movie():

  assert os.path.exists(MOVIE)
  cmdline = '%s --self-test=1' % (MOVIE)
  assert facebox.main(cmdline.split()) == 0

@utils.visioner_available
def test_face_detect_on_image():

  assert os.path.exists(IMAGE)
  cmdline = '%s --self-test=2' % (IMAGE)
  assert facebox.main(cmdline.split()) == 0

@utils.visioner_available
@utils.ffmpeg_found()
def test_keypoint_localization_on_movie():

  assert os.path.exists(MOVIE)
  cmdline = '%s --self-test=1' % (MOVIE)
  assert facepoints.main(cmdline.split()) == 0

@utils.visioner_available
def test_keypoint_localization_on_image():

  assert os.path.exists(IMAGE)
  cmdline = '%s --self-test=2' % (IMAGE)
  assert facepoints.main(cmdline.split()) == 0
