#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Mar 29 12:35:10 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""This example bob application produces a video output that shows the
Optical Flow (using an HSL mapping) of a given input video. You should pass the
input filename (movie) and the output filename (output) that will contain the
resulting movie. It is possible that you use the string "%(stem)s" to imply the
original filename stem (basename minus extension). Example:
"outdir/%(stem)s.avi"."""

import sys
import os
import optparse
import numpy
import tempfile
import shutil
import bob

def optflow_hs(movie, iterations, alpha, template, stop=0):
  """This method is the one you are interested, it shows how bob reads a
  video file and computes the optical flow using the Horn & Schunck method,
  saving the output as a new video in the output directory (using a template
  based on the original movie filename stem (base filename minus extension).
  
  The first flow is calculated from scratch setting the initial velocities in
  the width and height direction (U and V) to zero. The subsequent flows are
  calculated using the previous frame flow estimation.
  """

  tmpl_fill = {'stem': os.path.splitext(os.path.basename(movie))[0]}
  if template.find('%(stem)s') != -1: output = template % tmpl_fill
  else: output = template

  # Makes sure we don't overwrite the original file
  if (os.path.realpath(movie) == os.path.realpath(output)):
    raise RuntimeError("Input and output refer to the same file '%s'" % output)

  outputdir = os.path.dirname(output)

  # Creates the output directory if that does not exist
  if not os.path.exists(outputdir): os.makedirs(outputdir)

  # To read the input we use the VideoReader class and its iterability
  video = bob.io.VideoReader(movie)
  print("Loading", video.info)

  # The images for the optical flow computation must be grayscale
  previous = None
  
  # These are the output vectors from the flow computation
  u = numpy.ndarray((video.height, video.width), 'float64')
  v = numpy.ndarray((video.height, video.width), 'float64')
  
  # Creates the output video (frame rate by default)
  outvideo = bob.io.VideoWriter(output, video.height, video.width)

  print("Horn & Schunck Optical Flow: alpha = %.2f; iterations = %d" % \
      (alpha, iterations))
  flow = bob.ip.VanillaHornAndSchunckFlow((video.height, video.width))
  for k, frame in enumerate(video):
    if previous is None:
      # we need 2 images to compute the flow, if we are on the first iteration,
      # keep the image and defer the calculation until we have a second frame
      previous = bob.ip.rgb_to_gray(frame)
      continue

    # if you get to this point, we have two consecutive images
    current = bob.ip.rgb_to_gray(frame)
    flow(alpha, iterations, previous, current, u, v)
    
    # please note the HS algorithm output is as float64 and that the flow2hsv
    # method outputs in float32 (read respective documentations)
    float_rgb = bob.ip.flowutils.flow2hsv(u,v)
    outvideo.append((255.0*float_rgb).astype('uint8'))

    # reset the "previous" frame
    previous = current
    
    sys.stdout.write('.')
    sys.stdout.flush()

    if stop and k > stop: break #for testing purposes, only run a subset

  print("\nWrote %d frames to %s" % (k, output))

def main(user_input=None):

  import argparse
  
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument("movie", type=str, metavar='FILE',
      help="Input movie to be treated by this script")
  parser.add_argument("output", type=str, metavar='FILE',
      help="Output movie containing the HSV representation of the flow")

  parser.add_argument("-a", "--alpha",
      action="store", dest="alpha", default=2.0, type=float, metavar='FLOAT',
      help="Modifies the proportion of smoothness in the H&S algorithm (defaults to %(default)s)")
  parser.add_argument("-i", "--iterations",
      action="store", dest="iterations", default=1, type=int, metavar='FLOAT',
      help="Modifies the proportion of smoothness in the H&S algorithm (defaults to %(default)s)")

  # This option is not normally shown to the user...
  parser.add_argument("--self-test", action="store_true", dest="selftest",
      default=False, help=argparse.SUPPRESS)

  args = parser.parse_args(args=user_input)

  if args.selftest:
    # then we go into test mode, all input is preset
    outputdir = tempfile.mkdtemp()
    output = os.path.join(outputdir, "%(stem)s.avi")
    try:
      #1 iter. per cycle is faster
      optflow_hs(args.movie, 1, args.alpha, output, 10)
    finally:
      shutil.rmtree(outputdir)

  else:
    optflow_hs(args.movie, args.iterations, args.alpha, args.output)

  return 0
