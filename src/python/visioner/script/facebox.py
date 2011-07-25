#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 25 Jul 14:07:42 2011 

"""Detects faces on video or images, imprint bounding boxes or just do a
textual dump of such values. Bounding boxes are 4-tuples containing the upper
left corner x and y coordinates followed by the bounding box window width and
its height."""

__epilog__ = """Example usage:

1. Detect faces in a video

  $ %(prog)s myvideo.mov

2. Detect faces in an image

  $ %(prog)s myimage.jpg

3. Detect faces in a video, imprint results on an output video copy, shows
   optional verbose output

  $ %(prog)s --verbose myvideo.mov result.avi

4. Detect faces in an image, imprint results on output image

  $ %(prog)s myimage.jpg result.png
"""

import os
import sys
import time
import argparse
import torch
import tempfile #for package tests

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
  gray_buffer = torch.core.array.uint8_2(input.height, input.width)
  data = []
  total = 0
  if args.verbose:
    sys.stdout.write("Detecting (single) faces in %d frames from file %s" % \
        (input.numberOfFrames, args.input))
  for k in input:
    torch.ip.rgb_to_gray(k, gray_buffer)
    int16_buffer = gray_buffer.cast('int16')
    start = time.clock()
    detections = args.processor(int16_buffer)
    if args.verbose:
      sys.stdout.write('.')
      sys.stdout.flush()
    total += time.clock() - start
    if len(detections) == 0: best = None
    else: best = detections[0]
    data.append(best)

  if args.verbose: sys.stdout.write('\n')

  if args.verbose:
    print "Total localization/detection time was %.2f seconds" % total
    print " -> Per image/frame %.3f seconds" % (total/input.numberOfFrames)

  if not args.output:
    for k, det in enumerate(data):
      if not det: det = (k, 0, 0, 0, 0, -sys.float_info.max)
      else: det = (k, r(det[0]), r(det[1]), r(det[2]), r(det[3]), det[4])
      if args.dump_scores:
        sys.stdout.write("%d %d %d %d %d %.4e\n" % det)
      else:
        sys.stdout.write("%d %d %d %d %d\n" % det[:5])

  else: #use wants to record a video with the output
   
    if args.verbose:
      sys.stdout.write("Saving %d frames with detections to %s" % \
          (len(data), args.output))

    color = (255, 0, 0) #red
    orows = 2*(input.height/2)
    ocolumns = 2*(input.width/2)
    ov = torch.io.VideoWriter(args.output, orows, ocolumns, input.frameRate)
    for frame,bbox in zip(input,data):
      if bbox:
        bbox = [r(v) for v in bbox[:4]]
        # 3-pixels width box
        torch.ip.draw_box(frame, bbox[0], bbox[1], bbox[2], bbox[3], color)
        torch.ip.draw_box(frame, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2, 
            color)
        torch.ip.draw_box(frame, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2, 
            color)
      ov.append(frame[:,:orows,:ocolumns])
      if args.verbose:
        sys.stdout.write('.')
        sys.stdout.flush()

    if args.verbose: sys.stdout.write('\n')

def process_image_data(args):
  """Process any kind of image data"""

  if args.verbose: print "Loading file %s..." % args.input
  input = torch.core.array.load(args.input) #load the image

  if input.rank() == 3: #it is a color image
    graydata = torch.ip.rgb_to_gray(input).cast('int16')
  elif input.rank() == 2: #it is a gray-scale image
    graydata = input.cast('int16')

  start = time.clock()
  detections = args.processor(graydata)
  total = time.clock() - start
  if len(detections) == 0: data = None
  else: data = detections[0]
  if args.verbose:
    print "Total localization/detection time was %.3f seconds" % total

  if not args.output:
    
    if not data: data = (0, 0, 0, 0, -sys.float_info.max)
    else: data = (r(data[0]), r(data[1]), r(data[2]), r(data[3]), data[4])
    if args.dump_scores:
      sys.stdout.write("%d %d %d %d %.4e\n" % data)
    else:
      sys.stdout.write("%d %d %d %d\n" % data[:4])

  else: #use wants to record an image with the output

    if data:
      if input.rank() == 3: color = (255, 0, 0) #red
      else: color = 255
      bbox = [r(v) for v in data[:4]]
      torch.ip.draw_box(input, bbox[0], bbox[1], bbox[2], bbox[3], color)
      torch.ip.draw_box(input, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2,
          color)
      torch.ip.draw_box(input, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2,
          color)

    input.save(args.output)

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
  parser.add_argument("-t", "--threshold", metavar='FLOAT',
      type=float, dest='threshold', default=-sys.float_info.max,
      help="classifier threshold (defaults to %(default)s)")
  parser.add_argument("-d", "--dump-scores",
      default=False, action='store_true', dest='dump_scores',
      help="if set, also dump scores after every bounding box")
  parser.add_argument("-v", "--verbose", dest="verbose",
      default=False, action='store_true',
      help="enable verbose output")
  parser.add_argument("--self-test", metavar='INT', type=int, default=False,
      dest='selftest', help=argparse.SUPPRESS)

  args = parser.parse_args()

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
    args.dump_scores = True

  start = time.clock() 
  args.processor = torch.visioner.Detector(cmodel_file=args.cmodel,
      threshold=args.threshold)
  total = time.clock() - start

  if args.verbose:
    print "Model loading took %.2f seconds" % total

  is_video = (os.path.splitext(args.input)[1] in torch.io.video_extensions())

  if is_video:
    process_video_data(args)

  else:
    process_image_data(args)

  if args.selftest:
    os.unlink(args.output)

if __name__ == '__main__':
  main()
