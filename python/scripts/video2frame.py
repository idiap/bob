#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 27 Jul 2010 17:40:46 CEST 

"""Breaks a video file in a sequence of jpeg files.
"""

import sys, os
import torch

def main():
  if len(sys.argv) < 2  or len(sys.argv) > 3:
    print 'usage: %s <video-input> [output-dir]' % \
        (os.path.basename(sys.argv[0]))
    sys.exit(1)

  input = torch.ip.Video(sys.argv[1])
  buffer = torch.ip.Image(1, 1, 1) #forces gray-scale conversion
  output = None
  if len(sys.argv) >= 3: output = sys.argv[2] 
  else: output = os.path.splitext(os.path.basename(sys.argv[1]))[0]
  if not os.path.exists(output): os.makedirs(output)

  frame = 0
  obuf = torch.ip.Image()
  while (input.read(buffer)): 
    buffer.save(os.path.join(output, 'frame_%04d.jpg' % frame))
    frame += 1
  input.close()
  print 'Recorded %d frames in %s' % (frame, output)

if __name__ == '__main__':
  main()

