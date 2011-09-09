#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 27 Jul 2010 17:40:46 CEST 

"""This example Torch application produces a video output that shows the
Optical Flow (using an HSL mapping) of a given input video. You should pass the
input filename (movie) and the output filename (output) that will contain the
resulting movie. It is possible that you use the string "%(stem)s" to imply the
original filename stem (basename minus extension). Example:
"outdir/%(stem)s.avi"."""

import sys, os, optparse
import tempfile, shutil #for package tests
import torch

def optflowHS(movie, iterations, alpha, template, stop=0):
  """This method is the one you are interested, it shows how torch reads a
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
    raise RuntimeError, "Input and output refer to the same file '%s'" % output

  outputdir = os.path.dirname(output)

  # Creates the output directory if that does not exist
  if not os.path.exists(outputdir): os.makedirs(outputdir)

  # To read the input we use the VideoReader class and its iterability
  video = torch.io.VideoReader(movie)
  print "Loading", video.info

  # The images for the optical flow computation must be grayscale
  previous = None
  
  # These are the output vectors from the flow computation
  u = torch.core.array.float64_2(video.height, video.width)
  v = torch.core.array.float64_2(video.height, video.width)
  
  # Creates the output video (frame rate by default)
  outvideo = torch.io.VideoWriter(output, video.height, video.width)

  print "Horn & Schunck Optical Flow: alpha = %.2f; iterations = %d" % \
      (alpha, iterations)
  flow = torch.ip.VanillaHornAndSchunckFlow((video.height, video.width))
  for k, frame in enumerate(video):
    if previous is None:
      # we need 2 images to compute the flow, if we are on the first iteration,
      # keep the image and defer the calculation until we have a second frame
      previous = torch.ip.rgb_to_gray(frame)
      continue

    # if you get to this point, we have two consecutive images
    current = torch.ip.rgb_to_gray(frame)
    flow(alpha, iterations, previous, current, u, v)
    
    # please note the HS algorithm output is as float64 and that the flow2hsv
    # method outputs in float32 (read respective documentations)
    float_rgb = torch.ip.flowutils.flow2hsv(u,v)
    outvideo.append((255.0*float_rgb).cast('uint8'))

    # reset the "previous" frame
    previous = current
    
    sys.stdout.write('.')
    sys.stdout.flush()

    if stop and k > stop: break #for testing purposes, only run a subset

  print "\nWrote %d frames to %s" % (k, output)

if __name__ == '__main__':
  parser=optparse.OptionParser(usage="usage: %prog [options] <movie> <output>",
      description=__doc__)
  parser.add_option("-a", "--alpha",
      action="store", dest="alpha", default=2.0, type=float,
      help="Modifies the proportion of smoothness in the H&S algorithm (defaults to %default)",
      metavar="FLOAT")
  parser.add_option("-i", "--iterations",
      action="store", dest="iterations", default=1, type=int,
      help="Modifies the proportion of smoothness in the H&S algorithm (defaults to %default)",
      metavar="FLOAT")

  # This option is not normally shown to the user...
  parser.add_option("--test",
      action="store_true", dest="test", default=False,
      help=optparse.SUPPRESS_HELP)
      #help="if set, runs an internal verification test and erases any output")

  (options, args) = parser.parse_args()

  if options.test:
    # then we go into test mode, all input is preset
    packdir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    outputdir = tempfile.mkdtemp()
    movie = os.path.join(packdir, '..', 'io', 'test', 'data', 'test.mov')
    output = os.path.join(outputdir, "%(stem)s.avi")
    optflowHS(movie, 1, options.alpha, output, 10) #1 iter. per cycle is faster
    shutil.rmtree(outputdir)

  else:
    # a user is trying to execute the example, act normally
    if len(args) != 2:
      parser.error("requires 2 arguments (the movie path and the output template file name) -- read the help message!")

    optflowHS(args[0], options.iterations, options.alpha, args[1])

  sys.exit(0)
