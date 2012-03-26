#!/usr/bin/env python

"""Test data aquisition
"""

import os, sys
import unittest
import bob.io
from bob.daq import *
import tempfile

def get_tempfilename(prefix='bobtest_', suffix='.avi'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.unlink(name)
  return name

INPUT_VIDEO = 'test.mov'
OUTPUT_VIDEO = get_tempfilename('bobtest_daq_video', suffix='')

class DaqTest(unittest.TestCase):
  """Performs various data aquisition tests."""
  
  def test_VideoReaderCamera(self):
    video = bob.io.VideoReader(INPUT_VIDEO)

    pf = PixelFormat.RGB24
    fs = FrameSize(video.width, video.height)
    fi = FrameInterval(1, int(video.frameRate))

    camera = VideoReaderCamera(video)

    self.assertTrue(camera.get_supported_pixel_formats()[0] == pf)
    self.assertTrue(camera.get_supported_frame_sizes(pf)[0] == fs)
    self.assertTrue(camera.get_supported_frame_intervals(pf, fs)[0] == fi)

    if hasVisioner:
      fl = VisionerFaceLocalization()
    else:
      fl = NullFaceLocalization()
    
    controller = SimpleController()
    display = ConsoleDisplay()

    outputWriter = BobOutputWriter()
    outputWriter.set_output_dir(os.path.dirname(OUTPUT_VIDEO))
    outputWriter.set_output_name(os.path.basename(OUTPUT_VIDEO))
    outputWriter.open(video.width, video.height, int(video.frameRate))

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

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(DaqTest)
