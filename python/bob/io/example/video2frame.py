#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

import sys
import os
import tempfile
import shutil #for package tests
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
    bob.io.write(frame, template % i)
  sys.stdout.write('\n')
  print "Wrote %d frames to %s" % (i, outputdir)

def main(user_input=None):

  import argparse
  
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument("inputvideo", metavar='FILE',
      help="the name of the input video to treat")

  parser.add_argument("-d", "--output-dir", metavar='DIR',
      dest="outdir", default=os.curdir,
      help="if you want the output on a different directory, set this variable")

  # This option is not normally shown to the user...
  parser.add_argument("--self-test", action="store_true", dest="selftest",
      default=False, help=argparse.SUPPRESS)

  args = parser.parse_args(args=user_input)

  if args.selftest:
    # then we go into test mode, all input is preset
    outputdir = tempfile.mkdtemp()
    video2frame(args.inputvideo, outputdir)
    shutil.rmtree(outputdir)

  else:
    video2frame(args.inputvideo, os.path.realpath(args.outdir))

  return 0
