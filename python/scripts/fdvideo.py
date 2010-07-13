#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 13 Jul 2010 15:41:07 CEST 

"""Detects faces in videos
"""

import os, sys
import torch

def main():
  if len(sys.argv) == 1:
    print '%s - Detects faces in videos' % os.path.basename(sys.argv[0])
    print 'usage: %s <input-video> <facefinder-params> [<output-video>]' \
        % sys.argv[0]

  output_video = None
  if len(sys.argv) >= 4: output_video = sys.argv[3]

  input = torch.ip.Video(sys.argv[0])
  finder = torch.scanning.FaceFinder(sys.argv[1])
  buffer = torch.ip.Image(1, 1, 1) #forces gray-scale conversion

  frame = 0
  while (input.read(buffer)):
    frame += 1
    if not finder.process(buffer):
      raise RuntimeError, 'FaceFinder could not process image'
    detections = finder.getPatterns()

    if len(detections) > 1:
      print 'Detected multiple faces on frame %d'



if __name__ == '__main__':
  main()
