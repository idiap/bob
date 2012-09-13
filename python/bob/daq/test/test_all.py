#!/usr/bin/env python

"""Test data aquisition
"""

import os
import sys
import unittest
import tempfile
import pkg_resources
from nose.plugins.skip import SkipTest
import functools

def ffmpeg_found(test):
  '''Decorator to check if the FFMPEG is available before enabling a test'''

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      from ...io._io import VideoReader, VideoWriter
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest('FFMpeg was not available at compile time')

  return wrapper

def F(f, module=None):
  """Returns the test file on the "data" subdirectory"""
  if module is None:
    return pkg_resources.resource_filename(__name__, os.path.join('data', f))
  return pkg_resources.resource_filename('bob.%s.test' % module, 
      os.path.join('data', f))

from .. import *

def get_tempfilename(prefix='bobtest_', suffix='.avi'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.unlink(name)
  return name

INPUT_VIDEO = F('test.mov', 'io')
OUTPUT_VIDEO = get_tempfilename('bobtest_daq_video', suffix='')

class DaqTest(unittest.TestCase):
  """Performs various data aquisition tests."""
  
  @ffmpeg_found
  def test_VideoReaderCamera(self):
    from ...io import VideoReader
    video = VideoReader(INPUT_VIDEO)

    pf = PixelFormat.RGB24
    fs = FrameSize(video.width, video.height)
    fi = FrameInterval(1, int(video.frame_rate))

    camera = VideoReaderCamera(video)

    self.assertTrue(camera.get_supported_pixel_formats()[0] == pf)
    self.assertTrue(camera.get_supported_frame_sizes(pf)[0] == fs)
    self.assertTrue(camera.get_supported_frame_intervals(pf, fs)[0] == fi)

    fl = VisionerFaceLocalization()
    
    controller = SimpleController()
    display = ConsoleDisplay()

    outputWriter = BobOutputWriter()
    outputWriter.set_output_dir(os.path.dirname(OUTPUT_VIDEO))
    outputWriter.set_output_name(os.path.basename(OUTPUT_VIDEO))
    outputWriter.open(video.width, video.height, int(video.frame_rate))

    controller.add_controller_callback(fl)
    controller.add_controller_callback(display)
    controller.add_stoppable(camera)

    controller.set_output_writer(outputWriter)
    controller.recording_delay = 1
    controller.length = 3

    fl.add_face_localization_callback(display)
 
    camera.add_camera_callback(controller)

    self.assertTrue(camera.start() == 0)
    self.assertTrue(fl.start() == True)
    display.start()

    del outputWriter

    text_file = OUTPUT_VIDEO + ".txt"
    video_file = OUTPUT_VIDEO + ".avi"

    self.assertTrue(os.path.exists(video_file))
    self.assertTrue(os.path.exists(text_file))

    os.unlink(video_file)
    os.unlink(text_file)
