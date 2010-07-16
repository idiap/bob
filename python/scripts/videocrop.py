#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 13 Jul 2010 15:41:07 CEST 

"""Crops faces in videos
"""

import os, sys
import torch

def direction(a):
  """Returns the signal of the argument"""
  if a<0: return -1
  elif a==0: return 0
  else: return 1

class SmoothLocation:
  """This class smoothens the location found by the face detector to avoid too
  many variations in the face location.
  """
  def __init__(self):
    """Initalize the location smoother. "max" represents the maximum amount of
    pixel deslocation and expansion allowed for the window.
    """
    self.p = None
    self.move = 1
    self.every = 10 #frames
    self.now = 0

  def learn(self, p):
    """Learns from another pattern, what it should do."""
    if self.p is None:
      self.p = torch.scanning.Pattern(p.x, p.y, p.width, p.height)
      return
    elif self.now == self.every:
      self.p.x += (self.move * direction(p.x - self.p.x))
      self.p.y += (self.move * direction(p.y - self.p.y))
      self.p.width += (self.move * direction(p.width - self.p.width))
      self.p.height += (self.move * direction(p.height - self.p.height))
      self.now = 0
    else:
      self.now += 1

def main():
  if len(sys.argv) == 1:
    print '%s - Crops faces in videos' % os.path.basename(sys.argv[0])
    print 'usage: %s <input-video> <facefinder-params> <geometric-normalization-params> [<output-video>]' \
        % sys.argv[0]
    sys.exit(1)

  input = torch.ip.Video(sys.argv[1])
  output = None #created after we know the output image size...
  finder = torch.scanning.FaceFinder(sys.argv[2])
  geom_norm = torch.ip.ipGeomNorm(sys.argv[3])
  gt_file = torch.trainer.BoundingBoxGTFile()
  if not geom_norm.setGTFile(gt_file):
    raise RuntimeError, 'ipGeomNorm could not set GTFile'

  buffer = torch.ip.Image(1, 1, 1) #forces gray-scale conversion

  frame = 0
  skipped = 0
  smooth = SmoothLocation()

  while (input.read(buffer)):
    frame += 1
    if not finder.process(buffer):
      raise RuntimeError, 'FaceFinder could not process image'
    detections = finder.getPatterns()

    if len(detections) != 1:
      skipped += 1
      continue

    smooth.learn(detections[0])
    gt_file.load(smooth.p)
    #gt_file.load(detections[0])
    if not geom_norm.process(buffer):
      raise RuntimeError, 'ipGeomNorm could not process image'
    oi = geom_norm.getOutputImage(0)

    finder.getScanner().deleteAllROIs()
    if not finder.getScanner().addROIs(buffer, 0.3):
      raise RuntimeError, 'Scanner could not set ROIs'

    if len(sys.argv) >= 5: 
      if not output:
        output = torch.ip.Video(sys.argv[4], oi.getWidth(), oi.getHeight(),
            input.bitrate(), input.framerate(), input.gop())
      if not output.write(oi):
        raise RuntimeError, 'Video backend cannot write frame %d' % frame

  input.close()
  if output: output.close()

  print 'Single face detection in %.1f%% (%d) of %d frames' % \
      ((100.0*(frame-skipped))/frame, (frame-skipped), frame)

if __name__ == '__main__':
  main()
