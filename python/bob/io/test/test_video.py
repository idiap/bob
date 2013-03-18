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
from ...test import utils
import unittest
import numpy

# These are some global parameters for the test.
INPUT_VIDEO = utils.datafile('test.mov', sys.modules[__name__])
INPUT_H264_VIDEO = utils.datafile('test_h264.mov', sys.modules[__name__])
OUTPUT_VIDEO = utils.temporary_filename(suffix='.avi')

class VideoTest(unittest.TestCase):
  """Performs various combined read/write tests on video files"""
  
  @utils.ffmpeg_found()
  def test01_CanOpen(self):

    # This test opens and verifies some properties of the test video available.
    # It examplifies how to directly call the VideoReader and how to access
    # some of its properties.
    from .. import VideoReader
    v = VideoReader(INPUT_VIDEO)
    self.assertEqual(v.height, 240)
    self.assertEqual(v.width, 320)
    self.assertEqual(v.duration, 15000000) #microseconds
    self.assertEqual(v.number_of_frames, 375)
    self.assertEqual(len(v), 375)
    self.assertEqual(v.codec_name, 'mjpeg')

  def canReadImages(self, filename):
    
    # This test shows how you can read image frames from a VideoReader
    from .. import VideoReader
    v = VideoReader(filename)
    counter = 0
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
      counter += 1

    self.assertEqual(counter, len(v)) #we have gone through all frames

  @utils.ffmpeg_found()
  def test02_CanReadImages(self):
    self.canReadImages(INPUT_VIDEO)

  @utils.ffmpeg_found()
  @utils.codec_available('h264')
  def test02a_CanReadImagesH264(self):
    self.canReadImages(INPUT_H264_VIDEO)

  def canGetSpecificFrames(self, filename):

    # This test shows how to get specific frames from a VideoReader

    from .. import VideoReader
    v = VideoReader(filename)

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

  @utils.ffmpeg_found()
  def test03_CanGetSpecificFrames(self):

    self.canGetSpecificFrames(INPUT_VIDEO)

  @utils.ffmpeg_found()
  @utils.codec_available('h264')
  def test03a_CanGetSpecificFramesH264(self):

    self.canGetSpecificFrames(INPUT_H264_VIDEO)

  @utils.ffmpeg_found()
  def test04_CanWriteVideo(self):
    
    try:

      # This test reads all frames in sequence from a initial video and records
      # them into an output video, possibly transcoding it.
      from .. import VideoReader, VideoWriter
      iv = VideoReader(INPUT_VIDEO)
      ov = VideoWriter(OUTPUT_VIDEO, iv.height, iv.width)
      for k, frame in enumerate(iv): ov.append(frame)
     
      # We verify that both videos have similar properties
      self.assertEqual(len(iv), len(ov))
      self.assertEqual(iv.width, ov.width)
      self.assertEqual(iv.height, ov.height)
     
      ov.close() # forces close; see github issue #6
      del ov # trigger closing of the output video stream

      iv2 = VideoReader(OUTPUT_VIDEO)

      # We verify that both videos have similar frames
      counter = 0
      for orig, copied in zip(iv.__iter__(), iv2.__iter__()):
        diff = abs(orig.astype('float32')-copied.astype('float32'))
        m = numpy.mean(diff)
        self.assertTrue(m < 10) # average difference is less than 10 gray levels
        counter += 1

      self.assertEqual(counter, len(iv)) #we have gone through all frames
      
      del iv2 # triggers closing of the input video stream

    finally:
      os.unlink(OUTPUT_VIDEO)

  @utils.ffmpeg_found()
  def test05_CanUseArrayInterface(self):

    # This shows you can use the array interface to read an entire video
    # sequence in a single shot
    from .. import load, VideoReader
    array = load(INPUT_VIDEO)
    iv = VideoReader(INPUT_VIDEO)
   
    for frame_id, frame in zip(range(array.shape[0]), iv.__iter__()):
      self.assertTrue ( numpy.array_equal(array[frame_id,:,:,:], frame) )

  def canIterateOnTheSpot(self, filename):

    # This test shows how you can read image frames from a VideoReader created
    # on the spot
    from .. import VideoReader
    video = VideoReader(filename)
    counter = 0
    for frame in video:
      self.assertTrue(isinstance(frame, numpy.ndarray))
      self.assertEqual(len(frame.shape), 3)
      self.assertEqual(frame.shape[0], 3) #color-bands (RGB)
      self.assertEqual(frame.shape[1], 240) #height
      self.assertEqual(frame.shape[2], 320) #width
      counter += 1
    
    self.assertEqual(counter, len(video)) #we have gone through all frames

  @utils.ffmpeg_found()
  def test06_CanIterateOnTheSpot(self):

    self.canIterateOnTheSpot(INPUT_VIDEO)

  @utils.ffmpeg_found()
  @utils.codec_available('h264')
  def test06a_CanIterateOnTheSpotH264(self):

    self.canIterateOnTheSpot(INPUT_H264_VIDEO)

  def patternReadWrite(self, codec="", suffix=".avi"):
      
    # This test shows we can do a pattern encoding/decoding and get video
    # readout right

    from .. import VideoReader, VideoWriter
    from ..utils import generate_colors
    fname = utils.temporary_filename(suffix=suffix)
  
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
        outv = VideoWriter(fname, height, width, framerate, codec=codec)
      else:
        outv = VideoWriter(fname, height, width, framerate)
      orig = []
      for i in range(0, frames):
        #newframe = numpy.random.random_integers(0,255,(3,height,width)).astype('u8')
        newframe = generate_colors(height, width, i%width)
        outv.append(newframe)
        orig.append(newframe)
      outv.close()
      input = VideoReader(fname)
      reloaded = input.load()

      self.assertEqual( reloaded.shape[1:], orig[0].shape )
      self.assertEqual( len(reloaded), len(orig) )

      for i in range(len(reloaded)):
        diff = abs(reloaded[i].astype('float')-orig[i].astype('float'))
        m = numpy.mean(diff)
        self.assertTrue(m < 50.0) # TODO: too much compression loss

    finally:

      if os.path.exists(fname): os.unlink(fname)

  def patternReadTwice(self, codec="", suffix=".avi"):

    # This test shows if we can read twice the same video and get the 
    # same results all the time.

    from .. import load, VideoReader, VideoWriter
    from ..utils import generate_colors
    fname = utils.temporary_filename(suffix=suffix)
  
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
        outv = VideoWriter(fname, height, width, framerate, codec=codec)
      else:
        outv = VideoWriter(fname, height, width, framerate)
      orig = []
      for i in range(0, frames):
        #newframe = numpy.random.random_integers(0,255,(3,height,width)).astype('u8')
        newframe = generate_colors(height, width, i%width)
        outv.append(newframe)
        orig.append(newframe)
      outv.close()

      input1 = load(fname)
      input2 = load(fname)

      self.assertEqual( input1.shape, input2.shape )

      for i in range(len(input1)):
        diff = abs(input1[i].astype('float')-input2[i].astype('float'))
        m = numpy.mean(diff)
        self.assertTrue(m < 0.1)

    finally:

      if os.path.exists(fname): os.unlink(fname)

  @utils.ffmpeg_found()
  def test07_PatternReadWrite(self):
    self.patternReadWrite("")
    self.patternReadTwice("")
      
  @utils.ffmpeg_found()
  @utils.codec_available('mpeg4')
  def test08_PatternReadWrite_mpeg4(self):
    self.patternReadWrite("mpeg4")
    self.patternReadTwice("mpeg4")
      
  @utils.ffmpeg_found()
  @utils.codec_available('ffv1')
  def test09_PatternReadWrite_ffv1(self):
    self.patternReadWrite("ffv1")
    self.patternReadTwice("ffv1")
      
  @utils.ffmpeg_found()
  @utils.codec_available('h264')
  def test10_PatternReadWrite_h264(self):
    self.patternReadWrite("h264", ".mov")
    self.patternReadTwice("h264", ".mov")
