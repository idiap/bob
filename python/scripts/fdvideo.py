#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 13 Jul 2010 15:41:07 CEST 

"""Detects faces in videos
"""

import os, sys
import torch

def draw_rectangle(i, positions, line_width, color):
  """Draws a rectangle on an image in a certain position."""
  cimage = torch.ip.Image(i.getWidth(), i.getHeight(), 3)
  cimage.copyFromImage(i)
  for i in positions:
    for k in range(line_width):
      cimage.drawRect(i.x-k, i.y-k, i.width+(2*k), i.height+(2*k), color) 
  return cimage

def main():
  if len(sys.argv) == 1:
    print '%s - Detects faces in videos' % os.path.basename(sys.argv[0])
    print 'usage: %s <input-video> <facefinder-params> [<output-video>]' \
        % sys.argv[0]
    sys.exit(1)

  input = torch.ip.Video(sys.argv[1])
  finder = torch.scanning.FaceFinder(sys.argv[2])
  buffer = torch.ip.Image(1, 1, 1) #forces gray-scale conversion
  output = None
  if len(sys.argv) >= 4: output = torch.ip.Video(sys.argv[3], input)

  frame = 0
  while (input.read(buffer)):
    frame += 1
    if not finder.process(buffer):
      raise RuntimeError, 'FaceFinder could not process image'
    detections = finder.getPatterns()

    print 'frame: %d, detections: %d' % (frame, len(detections))
    for k in detections: print k

    finder.getScanner().deleteAllROIs()
    if not finder.getScanner().addROIs(buffer, 0.3):
      raise RuntimeError, 'Scanner could not set ROIs'

    if output:
      output.write(draw_rectangle(buffer, detections, 3, torch.ip.red))

  input.close()
  if output: output.close()

if __name__ == '__main__':
  main()
