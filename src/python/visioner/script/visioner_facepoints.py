#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 25 Jul 14:07:42 2011 

"""Detects faces and keypoints on video or images, imprint bounding boxes and
landmarks or just do a textual dump of such values. Bounding boxes are
4-tuples containing the upper left corner x and y coordinates followed by the
bounding box window width and its height. Keypoints are 2-tuples with (x,y)."""

__epilog__ = """Example usage:

1. Localize faces in a video

  $ %(prog)s myvideo.mov

2. Localize faces in an image

  $ %(prog)s myimage.jpg

3. Localize faces in a video, imprint results on an output video copy, shows
   optional verbose output

  $ %(prog)s --verbose myvideo.mov result.avi

4. Localize faces in an image, imprint results on output image

  $ %(prog)s myimage.jpg result.png
"""

import os
import sys
import time
import argparse
import torch
import tempfile #for package tests
import numpy

def testfile(path):
  """Computes the path to a test file"""
  d = os.path.join(os.path.dirname(__file__))
  return os.path.realpath(os.path.join(d, path))

def r(v):
  """Rounds the given float value to the nearest integer"""
  return int(round(v))

def process_video_data(args):
  """A more efficienty (memory-wise) way to process video data"""

  input = torch.io.VideoReader(args.input)
  gray_buffer = numpy.ndarray((input.height, input.width), 'uint8')
  data = []
  total = 0
  if args.verbose:
    sys.stdout.write("Detecting (single) faces in %d frames from file %s" % \
        (input.numberOfFrames, args.input))
  for k in input:
    torch.ip.rgb_to_gray(k, gray_buffer)
    int16_buffer = gray_buffer.astype('int16')
    start = time.clock()
    detections = args.processor(int16_buffer)
    total += time.clock() - start
    data.append(detections)
    if args.verbose:
      sys.stdout.write('.')
      sys.stdout.flush()

  if args.verbose: sys.stdout.write('\n')

  if args.verbose:
    print "Total localization time was %.2f seconds" % total
    print " -> Per image/frame %.3f seconds" % (total/input.numberOfFrames)

  if not args.output:

    for k, (bbox,points) in enumerate(data):
      if bbox:
        data = tuple([k] + [r(i) for i in bbox])
        sys.stdout.write("%d %d %d %d %d" % data)
        for p in points:
          sys.stdout.write(" %d %d" % (r(p[0]), r(p[1])))
        sys.stdout.write('\n')
      else:
        sys.stdout.write("%d 0 0 0 0\n" % k)

  else: #user wants to record a video with the output
   
    if args.verbose:
      sys.stdout.write("Saving %d frames with detections to %s" % \
          (len(data), args.output))

    red = (255, 0, 0)
    yellow = (255, 255, 0)
    orows = 2*(input.height/2)
    ocolumns = 2*(input.width/2)
    ov = torch.io.VideoWriter(args.output, orows, ocolumns, input.frameRate)

    for frame,(bbox,points) in zip(input,data):

      if bbox and sum(bbox):
        bbox = [r(v) for v in bbox]
        # 3-pixels width box
        torch.ip.draw_box(frame, bbox[0], bbox[1], bbox[2], bbox[3], red)
        torch.ip.draw_box(frame, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2, 
            red)
        torch.ip.draw_box(frame, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2, 
            red)

        for p in points:
          p = [r(v) for v in p]
          if sum(p): torch.ip.draw_cross(frame, p[0], p[1], 2, yellow)

      ov.append(frame[:,:orows,:ocolumns])

      if args.verbose:
        sys.stdout.write('.')
        sys.stdout.flush()

    if args.verbose: sys.stdout.write('\n')

def process_image_data(args):
  """Process any kind of image data"""

  if args.verbose: print "Loading file %s..." % args.input
  input = torch.io.load(args.input) #load the image

  if len(input.shape) == 3: #it is a color image
    graydata = torch.ip.rgb_to_gray(input).astype('int16')
  elif len(input.shape) == 2: #it is a gray-scale image
    graydata = input.astype('int16')

  start = time.clock()
  data = args.processor(graydata)
  total = time.clock() - start
  if args.verbose:
    print "Total localization time was %.3f seconds" % total

  bbox, points = data
  bbox = tuple([r(v) for v in bbox])

  if not args.output:
   
    if bbox:
      sys.stdout.write("%d %d %d %d" % bbox)
      for p in points:
        p = tuple([r(v) for v in p])
        sys.stdout.write(" %d %d" % p)
      sys.stdout.write('\n')
    else:
      sys.stdout.write("0 0 0 0\n")

  else: #user wants to record an image with the output

    if bbox and sum(bbox):
      
      if len(input.shape) == 3: 
        face = (255, 0, 0) #red
        cross = (255, 255, 0) #yellow
      else: 
        face = 255
        cross = 255

      torch.ip.draw_box(input, bbox[0], bbox[1], bbox[2], bbox[3], face)
      torch.ip.draw_box(input, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2,
          face)
      torch.ip.draw_box(input, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2,
          face)

      for p in points:
        p = tuple([r(v) for v in p])
        torch.ip.draw_cross(input, p[0], p[1], 2, cross)

    torch.io.save(input, args.output)

    if args.verbose:
      print "Output file (with detections, if any) saved at %s" % args.output

def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("input", metavar='FILE', type=str,
      help="the input filename")
  parser.add_argument("output", metavar='FILE', type=str, nargs='?',
      help="the output filename; if this filename is omitted, output for the detections is dumped in text format to the screen")
  parser.add_argument("-c", "--classification-model", metavar='FILE',
      type=str, dest="cmodel", default=None,
      help="use a classification model file different than the default")
  parser.add_argument("-l", "--localization-model", metavar='FILE',
      type=str, dest="lmodel", default=None,
      help="use a keypoint localization model file different than the default")
  parser.add_argument("-s", "--scan-levels", dest="scan_levels",
      default=0, type=int, metavar='INT>=0',
      help="scan levels (the higher, the faster - defaults to %(default)s)")
  parser.add_argument("-v", "--verbose", dest="verbose",
      default=False, action='store_true',
      help="enable verbose output")
  parser.add_argument("--self-test", metavar='INT', type=int, default=False,
      dest='selftest', help=argparse.SUPPRESS)

  args = parser.parse_args()

  if args.scan_levels < 0:
    parser.error("scanning levels have to be greater or equal 0")

  if args.selftest == 1:
    args.input = testfile('../../io/test/data/test.mov')
    (fd, filename) = tempfile.mkstemp('.avi', 'torchtest_')
    os.close(fd)
    os.unlink(filename)
    args.output = filename

  elif args.selftest == 2:
    args.input = testfile('../../ip/test/data/faceextract/test-faces.jpg')
    (fd, filename) = tempfile.mkstemp('.jpg', 'torchtest_')
    os.close(fd)
    os.unlink(filename)
    args.output = filename

  if args.selftest:
    args.verbose = True
    args.scan_levels = 10

  start = time.clock() 
  args.processor = torch.visioner.Localizer(cmodel_file=args.cmodel,
      lmodel_file=args.lmodel, scan_levels=args.scan_levels)
  total = time.clock() - start

  if args.verbose:
    print "Model loading took %.2f seconds" % total

  is_video = (os.path.splitext(args.input)[1] in ('.avi', '.h261', '.h263', '.h264', '.mov', '.m4v', '.mjpeg', '.mpeg', '.ogg', '.rawvideo'))

  if is_video:
    process_video_data(args)

  else:
    process_image_data(args)

  if args.selftest:
    os.unlink(args.output)

if __name__ == '__main__':
  main()
