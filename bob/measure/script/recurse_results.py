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


"""
This script parses through the given directory, collects all results of
verification experiments that are stored in file with the given file name.
It supports the split into development and test set of the data, as well as
ZT-normalized scores.

All result files are parsed and evaluated. For each directory, the following
information are given in columns:

  * The Equal Error Rate of the development set
  * The Equal Error Rate of the development set after ZT-Normalization
  * The Half Total Error Rate of the evaluation set
  * The Half Total Error Rate of the evaluation set after ZT-Normalization
  * The sub-directory where the scores can be found

The measure type of the development set can be changed to compute "HTER" or
"FAR" thresholds instead, using the --criterion option.
"""


import sys, os, bob
#from apport.hookutils import default



def get_args():
  """Parse the program options"""

  import argparse

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-d', '--devel-name', dest="dev", default="scores-dev", metavar="FILE",
                      help = "Name of the file containing the development scores")
  parser.add_argument('-e', '--eval-name', dest="eval", default="scores-eval", metavar="FILE",
                      help = "Name of the file containing the evaluation scores")
  parser.add_argument('-D', '--directory', default=".",
                      help = "The directory where the results should be collected from.")
  parser.add_argument('-n', '--nonorm-dir', dest="nonorm", default="nonorm", metavar = "DIR",
                      help = "Directory where the unnormalized scores are found")
  parser.add_argument('-z', '--ztnorm-dir', dest="ztnorm", default = "ztnorm", metavar = "DIR",
                      help = "Directory where the normalized scores are found")
  parser.add_argument('-s', '--sort', dest="sort", action='store_true',
                      help = "Sort the results")
  parser.add_argument('-k', '--sort-key', dest='key', default = 'dir', choices=['nonorm_dev','nonorm_eval','ztnorm_dev','ztnorm_eval','dir'],
                      help = "Sort the results accordign to the given key")
  parser.add_argument('-c', '--criterion', dest='criterion', default = 'HTER', choices=['HTER', 'EER', 'FAR'],
                      help = "Report Equal Rates (EER) rather than Half Total Error Rate (HTER)")

  parser.add_argument('-o', '--output', dest="output",
      help="Name of the output file that will contain the HTER scores")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  parser.add_argument('-p', '--parser', dest="parser", default="4column", metavar="NAME.FUNCTION",
      help="Name of a known parser or of a python-importable function that can parse your input files and return a tuple (negatives, positives) as blitz 1-D arrays of 64-bit floats. Consult the API of bob.measure.load.split_four_column() for details")

  # parse arguments
  args = parser.parse_args()

  # parse the score-parser
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


class Result:
  def __init__(self, dir, args):
    self.dir = dir
    self.m_args = args
    self.nonorm_dev = None
    self.nonorm_eval = None
    self.ztnorm_dev = None
    self.ztnorm_eval = None

  def __calculate__(self, dev_file, eval_file = None):
    dev_neg, dev_pos = self.m_args.parser(dev_file)

    # switch which threshold function to use;
    # THIS f***ing piece of code really is what python authors propose:
    threshold = {
      'EER'  : bob.measure.eer_threshold,
      'HTER' : bob.measure.min_hter_threshold,
      'FAR'  : bob.measure.far_threshold
    } [self.m_args.criterion](dev_neg, dev_pos)

    # compute far and frr for the given threshold
    dev_far, dev_frr = bob.measure.farfrr(dev_neg, dev_pos, threshold)
    dev_hter = (dev_far + dev_frr)/2.0

    if eval_file:
      eval_neg, eval_pos = self.m_args.parser(eval_file)
      eval_far, eval_frr = bob.measure.farfrr(eval_neg, eval_pos, threshold)
      eval_hter = (eval_far + eval_frr)/2.0
    else:
      eval_hter = None

    if self.m_args.criterion == 'FAR':
      return (dev_frr, eval_frr)
    else:
      return (dev_hter, eval_hter)

  def nonorm(self, dev_file, eval_file = None):
    (self.nonorm_dev, self.nonorm_eval) = self.__calculate__(dev_file, eval_file)

  def ztnorm(self, dev_file, eval_file = None):
    (self.ztnorm_dev, self.ztnorm_eval) = self.__calculate__(dev_file, eval_file)

  def __str__(self):
    str = ""
    for v in [self.nonorm_dev, self.ztnorm_dev, self.nonorm_eval, self.ztnorm_eval]:
      if v:
        val = "% 2.3f%%"%(v*100)
      else:
        val = "None"
      cnt = 16-len(val)
      str += " "*cnt + val
    str += "        %s"%self.dir
    return str[5:]


results = []

def add_results(args, nonorm, ztnorm = None):
  r = Result(os.path.dirname(nonorm).replace(os.getcwd()+"/", ""), args)
  print "Adding results from directory",r.dir
  # check if the results files are there
  dev_file = os.path.join(nonorm, args.dev)
  eval_file = os.path.join(nonorm, args.eval)
  if os.path.isfile(dev_file):
    if os.path.isfile(eval_file):
      r.nonorm(dev_file, eval_file)
    else:
      r.nonorm(dev_file)

  if ztnorm:
    dev_file = os.path.join(ztnorm, args.dev)
    eval_file = os.path.join(ztnorm, args.eval)
    if os.path.isfile(dev_file):
      if os.path.isfile(eval_file):
        r.ztnorm(dev_file, eval_file)
      else:
        r.ztnorm(dev_file)

  results.append(r)

def recurse(args, path):
  dir_list = os.listdir(path)

  # check if the score directories are included in the current path
  if args.nonorm in dir_list:
    if args.ztnorm in dir_list:
      add_results(args, os.path.join(path, args.nonorm), os.path.join(path, args.ztnorm))
    else:
      add_results(args, os.path.join(path, args.nonorm))

  for e in dir_list:
    real_path = os.path.join(path, e)
    if os.path.isdir(real_path):
      recurse(args, real_path)


def table():
  A = " "*2 + 'dev  nonorm'+ " "*5 + 'dev  ztnorm' + " "*6 + 'eval nonorm' + " "*4 + 'eval ztnorm' + " "*12 + 'directory\n'
  A += "-"*100+"\n"
  for r in results:
    A += str(r) + "\n"
  return A

def main():
  args = get_args()

  recurse(args, args.directory)

  if args.sort:
    import operator
    results.sort(key=operator.attrgetter(args.key))

  if args.self_test:
    _ = table()
  elif args.output:
    f = open(args.output, "w")
    f.writelines(table())
    f.close()
  else:
    print table()
