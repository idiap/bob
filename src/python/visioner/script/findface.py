#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 25 Jul 14:07:42 2011 

"""Detects faces on video or images, imprint bounding boxes, landmarks or just
do a textual dump of such values. Bounding boxes are 4-tuples containing the
upper left corner x and y coordinates followed by the bounding box window
width and its height. Keypoints are printed as 2-tuples."""

__epilog__ = """Example usage:

1. Detect faces in a video

  $ face_detect.py myvideo.mov

2. Detect faces in an image

  $ face_detect.py myimage.jpg

3. Detect faces in an image, imprint results on an image copy

  $ face_detect.py myimage.jpg result.png
"""

import os
import sys
import time
import argparse
import torch

def process_video_data(filename, processor, timeit):
  """A more efficienty (memory-wise) way to process video data"""

  input = torch.io.VideoReader(filename)
  gray_buffer = torch.core.array.uint8_2(input.height, input.width)
  retval = []
  total = 0
  for k in input:
    torch.ip.rgb_to_gray(k, gray_buffer)
    int16_buffer = gray_buffer.cast('int16')
    start = time.clock()
    retval.append(processor(int16_buffer))
    total += time.clock() - start

  if timeit:
    print "Total localization/detection time was %.2f seconds" % total
    print " -> Per image/frame %.3f seconds" % (total/input.numberOfFrames)

  return tuple(retval)

def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("input", metavar='FILE', type=str,
      help="the input filename")
  parser.add_argument("output", metavar='FILE', type=str, nargs='?',
      help="the output filename")
  parser.add_argument("-l", "--localization", dest="localize",
      default=False, action='store_true',
      help="enables full keypoint localization")
  parser.add_argument("-t", "--timeit", dest="timeit",
      default=False, action='store_true',
      help="enables printing timing information")

  args = parser.parse_args()

  start = time.clock()  
  Obj = None
  if args.localize:
    Obj = torch.visioner.Localizer()
  else:
    Obj = torch.visioner.Detector()
  total = time.clock() - start
  if args.timeit:
    print "Model loading took %.2f seconds" % total

  is_video = (os.path.splitext(args.input)[1] in torch.io.video_extensions())

  if is_video:
    # special treatment to save on memory utilisation
    data = process_video_data(args.input, Obj, args.timeit)
  else:
    input = torch.core.array.load(args.input) #load the image

    if input.rank() == 3: #it is a color image
      graydata = torch.ip.rgb_to_gray(input).cast('int16')
    elif input.rank() == 2: #it is a gray-scale image
      graydata = input.cast('int16')

    start = time.clock()
    data = (Obj(graydata),)
    total = time.clock() - start
    print "Total localization/detection time was %.3f seconds" % total

  # at this point, data is a list full of entries
  #import pdb; pdb.set_trace()

if __name__ == '__main__':
  main()
