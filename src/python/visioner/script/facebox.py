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

  $ face_detect.py myvideo.mov

2. Detect faces in an image

  $ face_detect.py myimage.jpg

3. Detect faces in a video, imprint results on an output video copy, shows
optional verbose output

  $ face_detect.py --verbose myvideo.mov result.avi

4. Detect faces in an image, imprint results on output image

  $ face_detect.py myimage.jpg result.png
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

def process_video_data(filename, processor, verbose, output):
  """A more efficienty (memory-wise) way to process video data"""

  input = torch.io.VideoReader(filename)
  gray_buffer = torch.core.array.uint8_2(input.height, input.width)
  data = []
  total = 0
  if verbose:
    sys.stdout.write("Detecting (single) faces in %d frames from file %s" % \
        (input.numberOfFrames, filename))
  for k in input:
    torch.ip.rgb_to_gray(k, gray_buffer)
    int16_buffer = gray_buffer.cast('int16')
    start = time.clock()
    detections = processor(int16_buffer)
    if verbose:
      sys.stdout.write('.')
      sys.stdout.flush()
    total += time.clock() - start
    if len(detections) == 0: best = None
    else: best = detections[0]
    data.append(best)

  if verbose: sys.stdout.write('\n')

  if verbose:
    print "Total localization/detection time was %.2f seconds" % total
    print " -> Per image/frame %.3f seconds" % (total/input.numberOfFrames)

  if not output:
    for k, det in enumerate(data):
      if det:
        det = [int(round(v)) for v in det]
        print k, det[0], det[1], det[2], det[3]
      else:
        print k, 0, 0, 0, 0

  else: #use wants to record a video with the output
   
    if verbose:
      sys.stdout.write("Saving %d frames with detections to %s" % \
          (len(data), output))

    color = (255, 0, 0) #red
    orows = 2*(input.height/2)
    ocolumns = 2*(input.width/2)
    ov = torch.io.VideoWriter(output, orows, ocolumns, input.frameRate)
    for frame,bbox in zip(input,data):
      if bbox:
        bbox = [int(round(v)) for v in bbox]
        # 3-pixels width box
        torch.ip.draw_box(frame, bbox[0], bbox[1], bbox[2], bbox[3], color)
        torch.ip.draw_box(frame, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2, 
            color)
        torch.ip.draw_box(frame, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2, 
            color)
      ov.append(frame[:,:orows,:ocolumns])
      if verbose:
        sys.stdout.write('.')
        sys.stdout.flush()

    if verbose: sys.stdout.write('\n')

def process_image_data(filename, processor, verbose, output):
  """Process any kind of image data"""

  input = torch.core.array.load(filename) #load the image

  if input.rank() == 3: #it is a color image
    graydata = torch.ip.rgb_to_gray(input).cast('int16')
  elif input.rank() == 2: #it is a gray-scale image
    graydata = input.cast('int16')

  start = time.clock()
  detections = processor(graydata)
  total = time.clock() - start
  if len(detections) == 0: data = None
  else: data = detections[0]
  if verbose:
    print "Total localization/detection time was %.3f seconds" % total

  if not output:
    
    if data:
      det = [int(round(v)) for v in data]
      print det[0], det[1], det[2], det[3]
    else:
      print 0, 0, 0, 0

  else: #use wants to record an image with the output

    if data:
      if input.rank() == 3: color = (255, 0, 0) #red
      else: color = 255
      bbox = [int(round(v)) for v in data]
      torch.ip.draw_box(input, bbox[0], bbox[1], bbox[2], bbox[3], color)
      torch.ip.draw_box(input, bbox[0]-1, bbox[1]-1, bbox[2]+2, bbox[3]+2,
          color)
      torch.ip.draw_box(input, bbox[0]+1, bbox[1]+1, bbox[2]-2, bbox[3]-2,
          color)

    input.save(output)

    if verbose:
      print "Output file (with detections, if any) saved at %s" % output

def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("input", metavar='FILE', type=str,
      help="the input filename")
  parser.add_argument("output", metavar='FILE', type=str, nargs='?',
      help="the output filename; if this filename is omitted, output for the detections is dumped in text format to the screen")
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
    args.verbose = True

  elif args.selftest == 2:
    args.input = testfile('../../ip/test/data/faceextract/test-faces.jpg')
    (fd, filename) = tempfile.mkstemp('.jpg', 'torchtest_')
    os.close(fd)
    os.unlink(filename)
    args.output = filename
    args.verbose = True

  start = time.clock()  
  processor = torch.visioner.Detector()
  total = time.clock() - start

  if args.verbose:
    print "Model loading took %.2f seconds" % total

  is_video = (os.path.splitext(args.input)[1] in torch.io.video_extensions())

  if is_video:
    process_video_data(args.input, processor, args.verbose, args.output)

  else:
    process_image_data(args.input, processor, args.verbose, args.output)

  if args.selftest:
    os.unlink(args.output)

if __name__ == '__main__':
  main()
