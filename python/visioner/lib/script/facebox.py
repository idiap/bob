#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon Jul 25 20:50:52 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Detects the most face-like object on video or images, imprint bounding boxes
or just do a textual dump of such values. Bounding boxes are 4-tuples
containing the upper left corner x and y coordinates followed by the bounding
box window width and its height."""

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
import bob
import tempfile #for package tests
import numpy

def r(v):
  """Rounds the given float value to the nearest integer"""
  return int(round(v))

def process_video_data(args):
  """A more efficienty (memory-wise) way to process video data"""

  input = bob.io.VideoReader(args.input)

  if args.start_frame < 0 or args.start_frame >= len(input):
    raise RuntimeError, "start frame has to set to a value between 0 and %d (inclusive)" % (len(input)-1,)

  if args.end_frame <= 0: args.end_frame = len(input)

  if args.end_frame < 0 or args.end_frame > len(input):
    raise RuntimeError, "end frame has to set to a value between 1 and %d (inclusive)" % (len(input),)

  if args.start_frame >= args.end_frame:
    raise RuntimeError, "start frame (%d) has to be smaller than end frame (%d)" % (args.start_frame, args.end_frame)

  gray_buffer = numpy.ndarray((input.height, input.width), 'uint8')
  data = []
  total = 0
  if args.verbose:
    sys.stdout.write("Detecting (single) faces in %d frames from file %s" % \
        (input.number_of_frames, args.input))

  valid_range = range(args.start_frame, args.end_frame)

  for i, k in enumerate(input):
    if i not in valid_range: continue

    bob.ip.rgb_to_gray(k, gray_buffer)
    start = time.clock()
    detection = args.processor(gray_buffer)
    if args.verbose:
      sys.stdout.write('.')
      sys.stdout.flush()
    total += time.clock() - start
    data.append(detection)

  if args.verbose: sys.stdout.write('\n')

  if args.verbose:
    print "Total detection time was %.2f seconds" % total
    print " -> Per image/frame %.3f seconds" % (total/len(valid_range))

  if not args.output:
    for k, det in enumerate(data):
      if not det: det = (k, 0, 0, 0, 0, -sys.float_info.max)
      else: det = (k, r(det[0]), r(det[1]), r(det[2]), r(det[3]), det[4])
      if args.dump_scores:
        sys.stdout.write("%d %d %d %d %d %.4e\n" % det)
      else:
        sys.stdout.write("%d %d %d %d %d\n" % det[:5])

  else: #user wants to record a video with the output
   
    if args.verbose:
      sys.stdout.write("Saving %d frames with detections to %s" % \
          (len(data), args.output))

    color = (255, 0, 0) #red
    orows = 2*(input.height/2)
    ocolumns = 2*(input.width/2)
    ov = bob.io.VideoWriter(args.output, orows, ocolumns, input.frame_rate)
    for frame,bbox in zip(input,data):
      if bbox and sum(bbox) != 0:
        bbox = [r(v) for v in bbox[:4]]
        # 3-pixels width box
        bob.ip.draw_box(frame, bbox[0], bbox[1], bbox[2], bbox[3], color)
        bob.ip.draw_box(frame, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2, 
            color)
        bob.ip.draw_box(frame, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2, 
            color)
      ov.append(frame[:,:orows,:ocolumns])
      if args.verbose:
        sys.stdout.write('.')
        sys.stdout.flush()

    if args.verbose: sys.stdout.write('\n')

def process_image_data(args):
  """Process any kind of image data"""

  if args.verbose: print "Loading file %s..." % args.input
  input = bob.io.load(args.input) #load the image

  if len(input.shape) == 3: #it is a color image
    graydata = bob.ip.rgb_to_gray(input)
  elif len(input.shape) == 2: #it is a gray-scale image
    graydata = input

  start = time.clock()
  data = args.processor(graydata)
  total = time.clock() - start
  if args.verbose:
    print "Total detection time was %.3f seconds" % total

  if not args.output:
    
    if not data: data = (0, 0, 0, 0, -sys.float_info.max)
    else: data = (r(data[0]), r(data[1]), r(data[2]), r(data[3]), data[4])
    if args.dump_scores:
      sys.stdout.write("%d %d %d %d %.4e\n" % data)
    else:
      sys.stdout.write("%d %d %d %d\n" % data[:4])

  else: #user wants to record an image with the output

    if data:
      if len(input.shape) == 3: color = (255, 0, 0) #red
      else: color = 255
      bbox = [r(v) for v in data[:4]]
      if sum(bbox):
        bob.ip.draw_box(input, bbox[0], bbox[1], bbox[2], bbox[3], color)
        bob.ip.draw_box(input, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2,
            color)
        bob.ip.draw_box(input, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2,
            color)

    bob.io.save(input, args.output)

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
      type=str, dest="model", default=None,
      help="use a classification model file different than the default")
  parser.add_argument("-d", "--dump-scores",
      default=False, action='store_true', dest='dump_scores',
      help="if set, also dump scores after every bounding box")
  parser.add_argument("-s", "--scanning-levels", dest="scan_levels",
      default=10, type=int, metavar='INT>=0',
      help="scan levels (the higher, the faster - defaults to %(default)s)")
  parser.add_argument("-v", "--verbose", dest="verbose",
      default=False, action='store_true',
      help="enable verbose output")
  parser.add_argument("-S", "--start-frame", dest='start_frame',
      type=int, default=0, 
      help="starts detection on the given frame (inclusive), in case you are treating videos (defaults to %(default)s")
  parser.add_argument("-E", "--end-frame", dest='end_frame',
      type=int, default=0, 
      help="ends detection on the given frame (exclusive), in case you are treating videos (give '0' to go through all frames; defaults to %(default)s")
  parser.add_argument("--self-test", metavar='INT', type=int, default=False,
      dest='selftest', help=argparse.SUPPRESS)

  args = parser.parse_args()

  if args.scan_levels < 0:
    parser.error("scanning levels have to be greater or equal 0")

  if args.selftest == 1:
    (fd, filename) = tempfile.mkstemp('.avi', 'bobtest_')
    os.close(fd)
    os.unlink(filename)
    args.output = filename
    args.start_frame = 0
    args.end_frame = 3 

  elif args.selftest == 2:
    (fd, filename) = tempfile.mkstemp('.jpg', 'bobtest_')
    os.close(fd)
    os.unlink(filename)
    args.output = filename

  if args.selftest:
    args.verbose = True
    args.dump_scores = True

  start = time.clock() 
  args.processor = bob.visioner.MaxDetector(model_file=args.model,
      scanning_levels=args.scan_levels)
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
