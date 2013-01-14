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

import os, sys
import tempfile
import pkg_resources
from nose.plugins.skip import SkipTest
import functools

# Here is a table of ffmpeg versions against libavcodec, libavformat and
# libavutil versions
from distutils.version import StrictVersion as SV
ffmpeg_versions = {
    '0.5.0':  [ SV('52.20.0'),   SV('52.31.0'),   SV('49.15.0')   ],
    '0.6.0':  [ SV('52.72.2'),   SV('52.64.2'),   SV('50.15.1')   ],
    '0.7.0':  [ SV('52.122.0'),  SV('52.110.0'),  SV('50.43.0')   ],
    '0.8.0':  [ SV('53.7.0'),    SV('53.4.0'),    SV('51.9.1')    ],
    '0.9.0':  [ SV('53.42.0'),   SV('53.24.0'),   SV('51.32.0')   ],
    '0.10.0': [ SV('53.60.100'), SV('53.31.100'), SV('51.34.101') ],
    '0.11.0': [ SV('54.23.100'), SV('54.6.100'),  SV('51.54.100') ],
    '1.0.0':  [ SV('54.59.100'), SV('54.29.104'), SV('51.54.100') ],
    }

def generate_pattern(height, width, counter):
  """Generates an image that serves as a test pattern for encoding/decoding and
  accuracy tests."""

  retval = numpy.ndarray((3, height, width), dtype='uint8') 

  # standard color test pattern
  w = width / 7; w2 = 2*w; w3 = 3*w; w4 = 4*w; w5 = 5*w; w6 = 6*w
  retval[0,:,0:w]   = 255; retval[1,:,0:w]   = 255; retval[2,:,0:w]   = 255;
  retval[0,:,w:w2]  = 255; retval[1,:,w:w2]  = 255; retval[2,:,w:w2]  = 0;
  retval[0,:,w2:w3] = 0;   retval[1,:,w2:w3] = 255; retval[2,:,w2:w3] = 255;
  retval[0,:,w3:w4] = 0;   retval[1,:,w3:w4] = 255; retval[2,:,w3:w4] = 0;
  retval[0,:,w4:w5] = 255; retval[1,:,w4:w5] = 0;   retval[2,:,w4:w5] = 255;
  retval[0,:,w5:w6] = 255; retval[1,:,w5:w6] = 0;   retval[2,:,w5:w6] = 0;
  retval[0,:,w6:]   = 0;   retval[1,:,w6:]  = 0;   retval[2,:,w6:]   = 255;

  # black bar by the end
  h = height - height/4
  retval[:,h:,:] = 0

  try:
    # text indicating the frame number 

    import Image, ImageFont, ImageDraw
    text = 'frame #%d' % counter
    font = ImageFont.load_default()
    (text_width, text_height) = font.getsize(text)
    img = Image.fromarray(retval.transpose(1,2,0))
    draw = ImageDraw.Draw(img)
    draw.text((5, 5*height/6), text, font=font, fill=(255,255,255))
    retval = numpy.asarray(img).transpose(2,0,1)

  except ImportError, e:
    pass

  return retval

def ffmpeg_found(version_geq=None):
  '''Decorator to check if a codec is available before enabling a test'''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      try:
        from .._io import version
        avcodec_inst= SV(version['FFmpeg']['avcodec'])
        avformat_inst= SV(version['FFmpeg']['avformat'])
        avutil_inst= SV(version['FFmpeg']['avutil'])
        if version_geq is not None:
          avcodec_req,avformat_req,avutil_req = ffmpeg_versions[version_geq]
          if avcodec_inst < avcodec_req:
            raise SkipTest('FFMpeg/libav version installed (%s) is smaller than required for this test (%s)' % (version['FFmpeg']['ffmpeg'], version_geq))
        return test(*args, **kwargs)
      except ImportError:
        raise SkipTest('FFMpeg was not available at compile time')

    return wrapper

  return test_wrapper

def codec_available(codec):
  '''Decorator to check if a codec is available before enabling a test'''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      d = bob.io.video_codecs()
      if d.has_key(codec) and d[codec]['encode'] and d[codec]['decode']:
        return test(*args, **kwargs)
      else:
        raise SkipTest('A functional codec for "%s" is not installed with FFmpeg' % codec)

    return wrapper

  return test_wrapper

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def get_tempfilename(prefix='bobtest_', suffix='.avi'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.unlink(name)
  return name

# These are some global parameters for the test.
INPUT_VIDEO = F('test.mov')
OUTPUT_VIDEO = get_tempfilename()

import unittest
import numpy
import bob

class VideoTest(unittest.TestCase):
  """Performs various combined read/write tests on video files"""
  
  @ffmpeg_found()
  def test01_CanOpen(self):

    # This test opens and verifies some properties of the test video available.
    # It examplifies how to directly call the VideoReader and how to access
    # some of its properties.
    v = bob.io.VideoReader(INPUT_VIDEO)
    self.assertEqual(v.height, 240)
    self.assertEqual(v.width, 320)
    self.assertEqual(v.duration, 15000000) #microseconds
    self.assertEqual(v.number_of_frames, 375)
    self.assertEqual(len(v), 375)
    self.assertEqual(v.codec_name, 'mjpeg')

  @ffmpeg_found()
  def test02_CanReadImages(self):

    # This test shows how you can read image frames from a VideoReader
    v = bob.io.VideoReader(INPUT_VIDEO)
    for frame in v:
      # Note that when you iterate, the frames are numpy.ndarray objects
      # So, you can use them as you please. The organization of the data
      # follows the other encoding systems in bob: (color-bands, height,
      # width).
      self.assertTrue(isinstance(frame, numpy.ndarray))
      self.assertEqual(len(frame.shape), 3)
      self.assertEqual(frame.shape[0], 3) #color-bands (RGB)
      self.assertEqual(frame.shape[1], 240) #height
      self.assertEqual(frame.shape[2], 320) #width

  @ffmpeg_found()
  def test03_CanGetSpecificFrames(self):

    # This test shows how to get specific frames from a VideoReader

    v = bob.io.VideoReader(INPUT_VIDEO)

    # get frame 27 (we start counting at zero)
    f27 = v[27]

    self.assertTrue(isinstance(f27, numpy.ndarray))
    self.assertEqual(len(f27.shape), 3)
    self.assertEqual(f27.shape, (3, 240, 320))

    # you can also use negative notation...
    self.assertTrue(isinstance(v[-1], numpy.ndarray))
    self.assertEqual(len(v[-1].shape), 3)
    self.assertEqual(v[-1].shape, (3, 240, 320))

    # get frames 18 a 30 (exclusive), skipping 3: 18, 21, 24, 27
    f18_30 = v[18:30:3]
    for k in f18_30:
      self.assertTrue(isinstance(k, numpy.ndarray))
      self.assertEqual(len(k.shape), 3)
      self.assertEqual(k.shape, (3, 240, 320))

    # the last frame in the sequence is frame 27 as you can check
    self.assertTrue( numpy.array_equal(f18_30[-1], f27) )

  @ffmpeg_found()
  def test04_CanWriteVideo(self):

    # This test reads all frames in sequence from a initial video and records
    # them into an output video, possibly transcoding it.
    iv = bob.io.VideoReader(INPUT_VIDEO)
    ov = bob.io.VideoWriter(OUTPUT_VIDEO, iv.height, iv.width)
    for k, frame in enumerate(iv): ov.append(frame)
   
    # We verify that both videos have similar properties
    self.assertEqual(len(iv), len(ov))
    self.assertEqual(iv.width, ov.width)
    self.assertEqual(iv.height, ov.height)
   
    ov.close() # forces close; see github issue #6
    del ov # trigger closing of the output video stream

    iv2 = bob.io.VideoReader(OUTPUT_VIDEO)

    # We verify that both videos have similar frames
    for orig, copied in zip(iv.__iter__(), iv2.__iter__()):
      diff = abs(orig.astype('float32')-copied.astype('float32'))
      m = numpy.mean(diff)
      self.assertTrue(m < 3.0) # average difference is less than 3 gray levels
    os.unlink(OUTPUT_VIDEO)

    del iv2 # triggers closing of the input video stream

  @ffmpeg_found()
  def test05_CanUseArrayInterface(self):

    # This shows you can use the array interface to read an entire video
    # sequence in a single shot
    array = bob.io.load(INPUT_VIDEO)
    iv = bob.io.VideoReader(INPUT_VIDEO)
   
    for frame_id, frame in zip(range(array.shape[0]), iv.__iter__()):
      self.assertTrue ( numpy.array_equal(array[frame_id,:,:,:], frame) )

  @ffmpeg_found()
  def test06_CanIterateOnTheSpot(self):

    # This test shows how you can read image frames from a VideoReader created
    # on the spot
    for frame in bob.io.VideoReader(INPUT_VIDEO):
      self.assertTrue(isinstance(frame, numpy.ndarray))
      self.assertEqual(len(frame.shape), 3)
      self.assertEqual(frame.shape[0], 3) #color-bands (RGB)
      self.assertEqual(frame.shape[1], 240) #height
      self.assertEqual(frame.shape[2], 320) #width

  def patternReadWrite(self, codec="", suffix=".avi"):
      
    # This test shows we can do a pattern encoding/decoding and get video
    # readout right

    fname = get_tempfilename(suffix=suffix)
  
    try:
      # Width and height should be powers of 2 as the encoded image is going 
      # to be approximated to the closest one, would not not be the case. 
      # In this case, the encoding is subject to more noise as the filtered,
      # final image that is encoded will contain added noise on the extra
      # borders.
      width = 128
      height = 128
      frames = 30
      framerate = 30 #Hz
      if codec:
        outv = bob.io.VideoWriter(fname, height, width, framerate, codec=codec)
      else:
        outv = bob.io.VideoWriter(fname, height, width, framerate)
      orig = []
      for i in range(0, frames):
        #newframe = numpy.random.random_integers(0,255,(3,height,width)).astype('u8')
        newframe = generate_pattern(height, width, i)
        outv.append(newframe)
        orig.append(newframe)
      outv.close()
      input = bob.io.VideoReader(fname)
      reloaded = input.load()

      self.assertEqual( reloaded.shape[1:], orig[0].shape )
      self.assertEqual( len(reloaded), len(orig) )

      for i in range(len(reloaded)):
        diff = abs(reloaded[i].astype('float')-orig[i].astype('float'))
        m = numpy.mean(diff)
        self.assertTrue(m < 5.0) # compression loss

    finally:

      if os.path.exists(fname): os.unlink(fname)

  def patternReadTwice(self, codec="", suffix=".avi"):

    # This test shows if we can read twice the same video and get the 
    # same results all the time.

    fname = get_tempfilename(suffix=suffix)
  
    try:
      # Width and height should be powers of 2 as the encoded image is going 
      # to be approximated to the closest one, would not not be the case. 
      # In this case, the encoding is subject to more noise as the filtered,
      # final image that is encoded will contain added noise on the extra
      # borders.
      width = 128
      height = 128
      frames = 30
      framerate = 30 #Hz
      if codec:
        outv = bob.io.VideoWriter(fname, height, width, framerate, codec=codec)
      else:
        outv = bob.io.VideoWriter(fname, height, width, framerate)
      orig = []
      for i in range(0, frames):
        #newframe = numpy.random.random_integers(0,255,(3,height,width)).astype('u8')
        newframe = generate_pattern(height, width, i)
        outv.append(newframe)
        orig.append(newframe)
      outv.close()

      input1 = bob.io.load(fname)
      input2 = bob.io.load(fname)

      self.assertEqual( input1.shape, input2.shape )

      for i in range(len(input1)):
        diff = abs(input1[i].astype('float')-input2[i].astype('float'))
        m = numpy.mean(diff)
        self.assertTrue(m < 0.1)

    finally:

      if os.path.exists(fname): os.unlink(fname)

  @ffmpeg_found()
  def test07_PatternReadWrite(self):
    self.patternReadWrite("")
    self.patternReadTwice("")
      
  @ffmpeg_found()
  @codec_available('mpeg4')
  def test08_PatternReadWrite_mpeg4(self):
    self.patternReadWrite("mpeg4")
    self.patternReadTwice("mpeg4")
      
  @ffmpeg_found()
  @codec_available('ffv1')
  def test09_PatternReadWrite_ffv1(self):
    self.patternReadWrite("ffv1")
    self.patternReadTwice("ffv1")
      
  @ffmpeg_found()
  @codec_available('h264')
  def test10_PatternReadWrite_h264(self):
    self.patternReadWrite("h264", ".mov")
    self.patternReadTwice("h264", ".mov")
