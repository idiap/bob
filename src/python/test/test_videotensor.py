#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 19 Jul 2010 13:39:20 CEST 

"""Tests the Torch::VideoTensor class
"""

import os, sys
import tempfile
import unittest
import torch

def get_tempfilename(prefix='torchtest_', suffix='.avi'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.unlink(name)
  return name

# These are some global parameters for the test.
OUTPUT_VIDEO = get_tempfilename()
OUTPUT_FILE = get_tempfilename(suffix='.tensor')
WIDTH = 600
HEIGHT = 400
FRAMES = 100
COLORS = (
          torch.ip.white, 
          torch.ip.blue, 
          torch.ip.red, 
          torch.ip.yellow, 
          torch.ip.green,
          torch.ip.cyan, 
          torch.ip.black, 
          torch.ip.pink
         )
FRAMERATE = 10
GROUP_OF_PICT = FRAMES/FRAMERATE

PATTERN = torch.ip.Image(WIDTH, HEIGHT, 3)
bar_WIDTH = WIDTH/len(COLORS)
for i, c in enumerate(COLORS):
  for k in range(bar_WIDTH):
    PATTERN.drawLine(k+(i*bar_WIDTH), 0, k+(i*bar_WIDTH), HEIGHT-1, c)

class VideoTensorTest(unittest.TestCase):
  """Performs various combined read/write tests on the Torch::VideoTensor object"""
  
  def test01_CanConstructFromScratch(self):
    vt = torch.ip.VideoTensor(WIDTH, HEIGHT, 3, FRAMES)
    for i in range(FRAMES): vt.setFrame(PATTERN, i)
    out_video = torch.ip.Video(OUTPUT_VIDEO, WIDTH, HEIGHT, 150000, FRAMERATE,
        GROUP_OF_PICT)
    self.assertEqual(vt.save(out_video), True)
    out_video.close()
    del out_video

  def test02_CanReadFromStandardVideoFile(self):
    video = torch.ip.Video(OUTPUT_VIDEO)
    #self.assertEqual(video.gop, GROUP_OF_PICT) #does not currently work!
    vt = torch.ip.VideoTensor(video, 1)
    self.assertEqual(vt.width, WIDTH)
    self.assertEqual(vt.height, HEIGHT)
    self.assertEqual(vt.planes, 1)
    self.assertEqual(vt.frames, FRAMES)
    #retrieves an image and compares it to the stock pattern
    image = torch.ip.Image(WIDTH, HEIGHT, 3)
    self.assertEqual(vt.getFrame(image, FRAMES/2), True)
    self.assertEqual(image.nplanes, 3) #image is black and white

  def test03_CanRecordOnTensorFile(self):
    vt = torch.ip.VideoTensor(WIDTH, HEIGHT, 3, FRAMES)
    for i in range(FRAMES): vt.setFrame(PATTERN, i)
    out_tensor = torch.core.TensorFile()
    out_tensor.openWrite(OUTPUT_FILE, vt)
    self.assertEqual(vt.save(out_tensor), True)
    out_tensor.close()

  def test04_CanReadFromTensorFile(self):
    video = torch.core.TensorFile()
    video.openRead(OUTPUT_FILE)
    vt = torch.ip.VideoTensor(video)
    self.assertEqual(vt.width, WIDTH)
    self.assertEqual(vt.height, HEIGHT)
    self.assertEqual(vt.planes, 3)
    self.assertEqual(vt.frames, FRAMES)
    #retrieves an image and compares it to the stock pattern
    image = torch.ip.Image(WIDTH, HEIGHT, 3)
    self.assertEqual(vt.getFrame(image, FRAMES/2), True)
    self.assertEqual(image.nplanes, 3) #image is color

  def test99_CleanUp(self):
    os.unlink(OUTPUT_VIDEO)
    os.unlink(OUTPUT_FILE)

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
