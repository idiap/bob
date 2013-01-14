#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed May 25 13:27:46 2011 +0200
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

"""This script applies a threshold to score file and reports error rates
"""

__epilog__ = """
Examples:

  1. Standard usage

     $ %(prog)s --scores=my-scores.txt --threshold=0.5
"""

import sys, os, bob

def apthres(neg, pos, thres):
  """Prints a single output line that contains all info for the threshold"""

  far, frr = bob.measure.farfrr(neg, pos, thres)
  hter = (far + frr)/2.0

  ni = neg.shape[0] #number of impostors
  fa = int(round(far*ni)) #number of false accepts
  nc = pos.shape[0] #number of clients
  fr = int(round(frr*nc)) #number of false rejects

  print("FAR : %.3f%% (%d/%d)" % (100*far, fa, ni))
  print("FRR : %.3f%% (%d/%d)" % (100*frr, fr, nc))
  print("HTER: %.3f%%" % (100*hter,))

def get_options(user_input):
  """Parse the program options"""

  usage = 'usage: %s [arguments]' % os.path.basename(sys.argv[0])

  import argparse
  parser = argparse.ArgumentParser(usage=usage,
      description=(__doc__ % {'prog': os.path.basename(sys.argv[0])}),
      epilog=(__epilog__ % {'prog': os.path.basename(sys.argv[0])}),
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--scores', dest="ifile", default=None,
      help="Name of the file containing the scores (defaults to %(default)s)",
      metavar="FILE")
  parser.add_argument('-t', '--threshold', dest='thres', default=None,
      type=float, help="The threshold value to apply", metavar="FLOAT")
  parser.add_argument('-p', '--parser', dest="parser", default="4column",
      help="Name of a known parser or of a python-importable function that can parse your input files and return a tuple (negatives, positives) as blitz 1-D arrays of 64-bit floats. Consult the API of bob.measure.load.split_four_column() for details", metavar="NAME.FUNCTION")
  
  # This option is not normally shown to the user...
  parser.add_argument("--self-test",
      action="store_true", dest="test", default=False, help=argparse.SUPPRESS)
      #help="if set, runs an internal verification test and erases any output")

  args = parser.parse_args(args=user_input)

  if args.test:
    # then we go into test mode, all input is preset
    args.thres = 0.0

  if args.ifile is None:
    parser.error("you should give an input score set with --scores")

  if args.thres is None:
    parser.error("you should give a threshold value with --threshold")

  #parse the score-parser
  if args.parser.lower() in ('4column', '4col'):
    args.parser = bob.measure.load.split_four_column
  elif args.parser.lower() in ('5column', '5col'):
    args.parser = bob.measure.load.split_five_column
  else: #try an import
    if args.parser.find('.') == -1:
      parser.error("parser module should be either '4column', '5column' or a valid python function identifier in the format 'module.function': '%s' is invalid" % args.parser)

    mod, fct = args.parser.rsplit('.', 2)
    import imp
    try:
      fp, pathname, description = imp.find_module(mod, ['.'] + sys.path)
    except Exception, e:
      parser.error("import error for '%s': %s" % (args.parser, e))

    try:
      pmod = imp.load_module(mod, fp, pathname, description)
      args.parser = getattr(pmod, fct)
    except Exception, e:
      parser.error("loading error for '%s': %s" % (args.parser, e))
    finally:
      fp.close()

  return args

def main(user_input=None):

  options = get_options(user_input)

  neg, pos = options.parser(options.ifile)
  apthres(neg, pos, options.thres)

  return 0
