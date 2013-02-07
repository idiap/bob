#!/usr/bin/env python

"""Test data aquisition
"""

import os
import sys
import unittest
from nose import SkipTest
from ...test import utils
from ...io import test as iotest

INPUT_VIDEO = utils.datafile('test.mov', iotest)
OUTPUT_VIDEO = utils.temporary_filename('bobtest_daq_video', suffix='.avi')

class DaqTest(unittest.TestCase):
  """Performs various data aquisition tests."""
  
  @utils.ffmpeg_found()
  def test_VideoReaderCamera(self):

    from ... import has_daq

    if not has_daq: raise SkipTest, "DAQ module was not compiled in"

    from .. import *
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
