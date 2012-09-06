#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed May 25 13:27:46 2011 +0200
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

"""This script computes the threshold following a certain minimization criteria
on the given input data."""

__epilog__ = """
Examples:

  1. Specify a different criteria (only mhter, mwer or eer accepted):

     $ %(prog)s --scores=dev.scores --criterium=mhter

  2. Calculate the threshold that minimizes the weither HTER for a cost of 0.4:

    $ %(prog)s --scores=dev.scores --criterium=mwer --cost=0.4

  3. Parse your input using a 5-column format

    $ %(prog)s --scores=dev.scores --parser=5column
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

def calculate(neg, pos, crit, cost):
  """Returns the threshold given a certain criteria"""

  if crit == 'eer':
    return bob.measure.eer_threshold(neg, pos)
  elif crit == 'mhter':
    return bob.measure.min_hter_threshold(neg, pos)

  # defaults to the minimum of the weighter error rate
  return bob.measure.min_weighted_error_rate_threshold(neg, pos, cost)

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
  parser.add_argument('-c', '--criterium', dest='crit', default='eer',
      choices=('eer', 'mhter', 'mwer'),
      help="The minimization criterium to use", metavar="CRITERIUM")
  parser.add_argument('-w', '--cost', dest='cost', default=0.5,
      type=float, help="The value w of the cost when minimizing using the minimum weighter error rate (mwer) criterium. This value is ignored for eer or mhter criteria.", metavar="FLOAT")
  parser.add_argument('-p', '--parser', dest="parser", default="4column",
      help="Name of a known parser or of a python-importable function that can parse your input files and return a tuple (negatives, positives) as blitz 1-D arrays of 64-bit floats. Consult the API of bob.measure.load.split_four_column() for details", metavar="NAME.FUNCTION")
  
  # This option is not normally shown to the user...
  parser.add_argument("--self-test",
      action="store_true", dest="test", default=False, help=argparse.SUPPRESS)
      #help="if set, runs an internal verification test and erases any output")

  args = parser.parse_args(args=user_input)

  if args.ifile is None:
    parser.error("you should give an input score set with --scores")

  if args.cost < 0.0 or args.cost > 1.0:
    parser.error("cost should lie between 0.0 and 1.0")

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
  t = calculate(neg, pos, options.crit, options.cost)
  print "Threshold:", t
  apthres(neg, pos, t)

  return 0
