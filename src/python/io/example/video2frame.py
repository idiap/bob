#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 27 Jul 2010 17:40:46 CEST 

"""This example Torch application breaks a video file in a sequence of jpeg
files."""

import sys, os, optparse
import tempfile, shutil #for package tests
import torch

def video2frame(movie, outputdir):
  """This method is the one you are interested, it shows how torch reads a
  video file and how to dump each frame as a separate image file."""

  # Creates the output directory if that does not exist
  if not os.path.exists(outputdir): os.makedirs(outputdir)

  # A template to save the frames to
  template = os.path.join(outputdir, "frame_%04d.jpg")

  # To read the input we use the VideoReader class and its iterability
  v = torch.io.VideoReader(movie)
  print "Loading", v.info
  for i, frame in enumerate(v):
    sys.stdout.write('.')
    sys.stdout.flush()
    torch.io.Array(frame).save(template % i)
  sys.stdout.write('\n')
  print "Wrote %d frames to %s" % (i, outputdir)

if __name__ == '__main__':
  parser = optparse.OptionParser(usage="usage: %prog [options] <movie>",
      description=__doc__)
  parser.add_option("-d", "--output-dir",
      action="store", dest="outdir", default=os.curdir,
      help="if you want the output on a different directory, set this variable",
      metavar="DIR")

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
    movie = os.path.join(packdir, 'test', 'data', 'test.mov')
    video2frame(movie, outputdir)
    shutil.rmtree(outputdir)

  else:
    # a user is trying to execute the example, act normally
    if len(args) != 1:
      parser.error("can only accept 1 argument (the movie path)")

    video2frame(args[0], os.path.realpath(options.outdir))

  sys.exit(0)
