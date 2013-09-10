#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
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

"""Runs some video tests
"""

import os
import sys
import numpy
from ...test import utils as testutils
from .. import supported_videowriter_formats
from ..utils import color_distortion, frameskip_detection, quality_degradation

# These are some global parameters for the test.
INPUT_VIDEO = testutils.datafile('test.mov', __name__)
SUPPORTED = supported_videowriter_formats()

@testutils.ffmpeg_found()
def test_can_use_array_interface():

  # This shows you can use the array interface to read an entire video
  # sequence in a single shot
  from .. import load, VideoReader
  array = load(INPUT_VIDEO)
  iv = VideoReader(INPUT_VIDEO)

  for frame_id, frame in zip(range(array.shape[0]), iv.__iter__()):
    assert numpy.array_equal(array[frame_id,:,:,:], frame)

@testutils.ffmpeg_found()
def test_can_iterate():

  # This test shows how you can read image frames from a VideoReader created
  # on the spot
  from .. import VideoReader
  video = VideoReader(INPUT_VIDEO)
  counter = 0
  for frame in video:
    assert isinstance(frame, numpy.ndarray)
    assert len(frame.shape) == 3
    assert frame.shape[0] == 3 #color-bands (RGB)
    assert frame.shape[1] == 240 #height
    assert frame.shape[2] == 320 #width
    counter += 1

  assert counter == len(video) #we have gone through all frames

@testutils.ffmpeg_found()
def check_format_codec(function, shape, framerate, format, codec, maxdist):

  length, height, width = shape
  fname = testutils.temporary_filename(suffix='.%s' % format)

  try:
    orig, framerate, encoded = function(shape, framerate, format, codec, fname)
    reloaded = encoded.load()

    # test number of frames is correct
    assert len(orig) == len(encoded), "original length %d != %d encoded" % (len(orig), len(encoded))
    assert len(orig) == len(reloaded), "original length %d != %d reloaded" % (len(orig), len(reloaded))

    # test distortion patterns (quick sequential check)
    dist = []
    for k, of in enumerate(orig):
      dist.append(abs(of.astype('float64')-reloaded[k].astype('float64')).mean())
    assert max(dist) <= maxdist, "max(distortion) %g > %g allowed" % (max(dist), maxdist)

    # assert we can randomly access any frame (choose 3 at random)
    for k in numpy.random.randint(length, size=(3,)):
      rdist = abs(orig[k].astype('float64')-encoded[k].astype('float64')).mean()
      assert rdist <= maxdist, "distortion(frame[%d]) %g > %g allowed" % (k, rdist, maxdist)

    # make sure that the encoded frame rate is not off by a big amount
    assert abs(framerate - encoded.frame_rate) <= (1.0/length), "reloaded framerate %g differs from original %g by more than %g" % (encoded.frame_rate, framerate, 1.0/length)

  finally:

    if os.path.exists(fname): os.unlink(fname)

def test_format_codecs():

  length = 30
  width = 128
  height = 128
  framerate = 30.
  shape = (length, height, width)
  methods = dict(
      frameskip = frameskip_detection,
      color     = color_distortion,
      noise     = quality_degradation,
      )

  # distortion patterns for specific codecs
  distortions = dict(
      # we require high standards by default
      default    = dict(frameskip=0.1,  color=8.5,  noise=45.),

      # high-quality encoders
      zlib       = dict(frameskip=0.0,  color=0.0, noise=0.0),
      ffv1       = dict(frameskip=0.05, color=9.,  noise=46.),
      vp8        = dict(frameskip=0.3,  color=9.0, noise=65.),
      libvpx     = dict(frameskip=0.3,  color=9.0, noise=65.),
      h264       = dict(frameskip=0.4,  color=8.5, noise=50.),
      libx264    = dict(frameskip=0.4,  color=8.5, noise=50.),
      theora     = dict(frameskip=0.5,  color=9.0, noise=70.),
      libtheora  = dict(frameskip=0.5,  color=9.0, noise=70.),
      mpeg4      = dict(frameskip=1.0,  color=9.0, noise=55.),

      # older, but still good quality encoders
      mjpeg      = dict(frameskip=1.2,  color=8.5, noise=50.),
      mpegvideo  = dict(frameskip=1.3,  color=8.5, noise=55.),
      mpeg2video = dict(frameskip=1.3,  color=8.5, noise=55.),
      mpeg1video = dict(frameskip=1.4,  color=9.0, noise=50.),

      # low quality encoders - avoid using - available for compatibility
      wmv2       = dict(frameskip=3.0,  color=10., noise=50.),
      wmv1       = dict(frameskip=2.5,  color=10., noise=50.),
      msmpeg4    = dict(frameskip=5.,   color=10., noise=50.),
      msmpeg4v2  = dict(frameskip=5.,   color=10., noise=50.),
      )

  # some exceptions
  if testutils.ffmpeg_version_lessthan('0.6'):
    distortions['ffv1']['frameskip'] = 0.55
    distortions['mpeg1video']['frameskip'] = 1.5
    distortions['mpegvideo']['color'] = 9.0
    distortions['mpegvideo']['frameskip'] = 1.4
    distortions['mpeg2video']['color'] = 9.0
    distortions['mpeg2video']['frameskip'] = 1.4


  for format in SUPPORTED:
    for codec in SUPPORTED[format]['supported_codecs']:
      for method in methods:
        check_format_codec.description = "%s.test_%s_format_%s_codec_%s" % (__name__, method, format, codec)
        distortion = distortions.get(codec, distortions['default'])[method]
        yield check_format_codec, methods[method], shape, framerate, format, codec, distortion

@testutils.ffmpeg_found()
def check_user_video(format, codec, maxdist):

  from .. import VideoReader, VideoWriter
  fname = testutils.temporary_filename(suffix='.%s' % format)
  MAXLENTH = 10 #use only the first 10 frames

  try:

    orig_vreader = VideoReader(INPUT_VIDEO)
    orig = orig_vreader[:MAXLENTH]
    (olength, _, oheight, owidth) = orig.shape
    assert len(orig) == MAXLENTH, "original length %d != %d MAXLENTH" % (len(orig), MAXLENTH)

    # encode the input video using the format and codec provided by the user
    outv = VideoWriter(fname, oheight, owidth, orig_vreader.frame_rate,
        codec=codec, format=format)
    for k in orig: outv.append(k)
    outv.close()

    # reload from saved file
    encoded = VideoReader(fname)
    reloaded = encoded.load()

    # test number of frames is correct
    assert len(orig) == len(encoded), "original length %d != %d encoded" % (len(orig), len(encoded))
    assert len(orig) == len(reloaded), "original length %d != %d reloaded" % (len(orig), len(reloaded))

    # test distortion patterns (quick sequential check)
    dist = []
    for k, of in enumerate(orig):
      dist.append(abs(of.astype('float64')-reloaded[k].astype('float64')).mean())
    assert max(dist) <= maxdist, "max(distortion) %g > %g allowed" % (max(dist), maxdist)

    # make sure that the encoded frame rate is not off by a big amount
    assert abs(orig_vreader.frame_rate - encoded.frame_rate) <= (1.0/MAXLENTH), "original video framerate %g differs from encoded %g by more than %g" % (encoded.frame_rate, framerate, 1.0/MAXLENTH)

  finally:

    if os.path.exists(fname): os.unlink(fname)

def test_user_video():

  # distortion patterns for specific codecs
  distortions = dict(
      # we require high standards by default
      default    = 1.5,

      # high-quality encoders
      zlib       = 0.0,
      ffv1       = 1.7,
      vp8        = 2.7,
      libvpx     = 2.7,
      h264       = 2.5,
      libx264    = 2.5,
      theora     = 2.0,
      libtheora  = 2.0,
      mpeg4      = 2.3,

      # older, but still good quality encoders
      mjpeg      = 1.8,
      mpegvideo  = 2.3,
      mpeg2video = 2.3,
      mpeg1video = 2.3,

      # low quality encoders - avoid using - available for compatibility
      wmv2       = 2.3,
      wmv1       = 2.3,
      msmpeg4    = 2.3,
      msmpeg4v2  = 2.3,
      )

  for format in SUPPORTED:
    for codec in SUPPORTED[format]['supported_codecs']:
      check_user_video.description = "%s.test_user_video_format_%s_codec_%s" % (__name__, format, codec)
      distortion = distortions.get(codec, distortions['default'])
      yield check_user_video, format, codec, distortion
