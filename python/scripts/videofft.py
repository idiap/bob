#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 27 Jul 2010 12:19:15 CEST 

"""This script will compute the fft of a video stream
"""

import sys, os
import torch

def asimage(t):
  """Transforms the FFT into an image (ignore complex part)"""
  n = torch.ip.Image(t.width, t.height, 3)
  amp = torch.ip.Image()
  amp.select(t, 2, 0) #gets the FFT amplitude
  for i in range(3):
    tmp = torch.ip.Image()
    tmp.select(n, 2, i)
    tmp.copy(amp)
  return n

def main():
  if len(sys.argv) < 2  or len(sys.argv) > 3:
    print 'usage: %s <video-input> [output-video]' % \
        (os.path.basename(sys.argv[0]))
    sys.exit(1)

  operator = torch.sp.spDCT()

  input = torch.ip.Video(sys.argv[1])
  buffer = torch.ip.Image(1, 1, 1) #forces gray-scale conversion
  output = None
  if len(sys.argv) >= 3: output = torch.ip.Video(sys.argv[2], input)

  frame = 0
  obuf = torch.ip.Image()
  while (input.read(buffer)):
    frame += 1
    if not operator.processImage(buffer, obuf):
      print 'Warning: frame %d not processed' % frame
    else:
      #amp = asimage(obuf)
      if output: output.write(obuf)

  input.close()
  if output: output.close()

if __name__ == '__main__':
  main()
