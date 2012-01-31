#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Sep 8 18:27:42 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

"""This example bob application produces a video output that shows the
Gradient Flow (using an HSL mapping) of a given input video. You should pass
the input filename (movie) and the output filename (output) that will contain
the resulting movie. It is possible that you use the string "%(stem)s" to imply
the original filename stem (basename minus extension). Example:
"outdir/%(stem)s.avi"."""

import sys, os, optparse
import tempfile, shutil #for package tests
import bob

def eval_gradient(movie, gradtype, template):
  """This method is the one you are interested, it shows how bob reads a
  video file and computes the gradient flow using either forward or central
  differences, saving the output as a new video in the output directory (using
  a template based on the original movie filename stem (base filename minus
  extension).
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
  video = bob.io.VideoReader(movie)
  print "Loading", video.info

  # The images for the optical flow computation must be grayscale
  previous = None
  
  # These are the output vectors from the flow computation
  u = bob.core.array.float64_2(video.height, video.width)
  v = bob.core.array.float64_2(video.height, video.width)
  
  # Creates the output video (frame rate by default)
  outvideo = bob.io.VideoWriter(output, video.height, video.width)

  if gradtype == 'forward':
    print "Computing Forward Spatio-Temporal Gradient (size 2)"
    grad = bob.ip.ForwardGradient((video.height, video.width))
  else:
    print "Computing Central Spatio-Temporal Gradient (Sobel Filter)"
    grad = bob.ip.CentralGradient((video.height, video.width))

  for k, frame in enumerate(video):

    if gradtype == 'forward':
      if previous is None:
        # Need 2 consecutive images to calculate the forward flow
        previous = [bob.ip.rgb_to_gray(frame).convert('float64',
            destRange=(0.,1.))]
        continue

    if gradtype == 'central':
      # Need 3 consecutive images to calculate the central flow
      if previous is None:
        previous = [bob.ip.rgb_to_gray(frame).convert('float64',
          destRange=(0.,1.))]
        continue
      elif previous and len(previous) == 1:
        previous.append(bob.ip.rgb_to_gray(frame).convert('float64',
          destRange=(0.,1.)))
        continue

    # if you get to this point, we have two/three consecutive images
    current = bob.ip.rgb_to_gray(frame).convert('float64', destRange=(0.,1.))
    args = previous + [current, u, v]
    grad(*args)
    
    # please note the algorithm output is as float64 and that the flow2hsv
    # method outputs in float32 (read respective documentations)
    rgb = bob.ip.flowutils.flow2hsv(u,v).convert('uint8', sourceRange=(0.,1.))
    outvideo.append(rgb)

    # reset the "previous" frame
    if gradtype == 'forward':
      previous[0] = current
    elif gradtype == 'central':
      previous[0] = previous[1]
      previous[1] = current
    
    sys.stdout.write('.')
    sys.stdout.flush()

  print "\nWrote %d frames to %s" % (k, output)

if __name__ == '__main__':
  parser=optparse.OptionParser(usage="usage: %prog [options] <movie> <output>",
      description=__doc__)
  parser.add_option("-t", "--type", choices=('central', 'forward'),
      default='central', dest='gtype', help='Defines the type of gradient to apply; options are central or forward (defaults to %default)')

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

    eval_gradient(args[0], options.gtype, args[1])

  sys.exit(0)
