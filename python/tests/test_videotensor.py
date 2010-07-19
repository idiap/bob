#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 19 Jul 2010 13:39:20 CEST 

"""Tests the Torch::VideoTensor class
"""

# These are some global parameters for the test.
INPUT_VIDEO = '/idiap/group/vision/visidiap/databases/banca/english/videos/1001_f_g1_04_1001_en.avi'
OUTPUT_VIDEO = 'out.avi'

import unittest
import torch

class VideoTensorTest(unittest.TestCase):
  """Performs various combined read/write tests on the Torch::VideoTensor object"""
  
  def test01_CanConstructFromScratch(self):
    width = 600
    height = 400
    frames = 100
    vt = torch.ip.VideoTensor(width, height, 3, frames)
    pattern = torch.ip.Image(width, height, 3)
    pattern.save("start.jpg")
    colors = (
              torch.ip.white, 
              torch.ip.blue, 
              torch.ip.red, 
              torch.ip.yellow, 
              #torch.ip.green,
              #torch.ip.cyan, 
              #torch.ip.black, 
              #torch.ip.pink
             )
    bar_width = width/len(colors)
    for i, c in enumerate(colors):
      for k in range(bar_width):
        pattern.drawLine(k+(i*bar_width), 0, k+(i*bar_width), height-1, c)
    pattern.save("test.jpg")
    import sys; exit(1)
    for i in range(100): vt.setFrame(pattern, i)
    out_video = torch.ip.Video(OUTPUT_VIDEO, width, height, 150000, 20, 5)
    self.assertEqual(vt.save(out_video), True)
    out_video.close()

  def test02_CanCopyUsingTensor(self):
    input_video = torch.ip.Video(INPUT_VIDEO)

if __name__ == '__main__':
  import sys
  sys.argv.append('-v')
  unittest.main()

