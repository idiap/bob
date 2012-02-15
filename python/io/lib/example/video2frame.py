#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
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

"""This example bob application breaks a video file in a sequence of jpeg
files."""

import sys, os, optparse
import tempfile, shutil #for package tests
import bob

def video2frame(movie, outputdir):
  """This method is the one you are interested, it shows how bob reads a
  video file and how to dump each frame as a separate image file."""

  # Creates the output directory if that does not exist
  if not os.path.exists(outputdir): os.makedirs(outputdir)

  # A template to save the frames to
  template = os.path.join(outputdir, "frame_%04d.jpg")

  # To read the input we use the VideoReader class and its iterability
  v = bob.io.VideoReader(movie)
  print "Loading", v.info
  for i, frame in enumerate(v):
    sys.stdout.write('.')
    sys.stdout.flush()
    bob.io.Array(frame).save(template % i)
  sys.stdout.write('\n')
  print "Wrote %d frames to %s" % (i, outputdir)

def main():

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
