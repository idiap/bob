#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 02 Aug 2012 14:31:58 CEST 
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

"""Trains a new boosted classifier based on LBP-like features from scratch.
"""

import os
import sys
import bob
import time

def as_parameter(args):
  """Creates a parameter structured copying data that are defined both 'args'
  and in 'param', from 'args'.
  """
  retval = bob.visioner.param()  
  for prop in [k for k in dir(args) if k[0] != '_']:
    if hasattr(retval, prop): setattr(retval, prop, getattr(args, prop))
  return retval

def main():

  import argparse

  parser = argparse.ArgumentParser(description=__doc__, 
      formatter_class=argparse.RawDescriptionHelpFormatter)

  defp = bob.visioner.param() # all defaults so we do DRY

  parser.add_argument("-r", "--rows", metavar='INT', type=int, dest='rows',
      default=defp.rows, help=bob.visioner.param.rows.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-c", "--columns", metavar='INT', type=int, dest='cols',
      default=defp.cols, help=bob.visioner.param.cols.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-s", "--seed", metavar='INT', type=int, dest='seed',
      default=defp.seed, help=bob.visioner.param.seed.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-b", "--label", metavar='LABEL', nargs='+', type=str,
      dest='labels', default=[], help=bob.visioner.param.labels.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-l", "--loss-type", metavar='LOSS', type=str,
      dest='loss', choices=bob.visioner.LOSSES, default=defp.loss,
      help=bob.visioner.param.loss.__doc__ + " (options: %s; default: %%(default)s)" % '|'.join(bob.visioner.LOSSES))
  parser.add_argument("-L", "--loss-parameter", metavar='FLOAT', type=float,
      dest='loss_parameter', default=defp.loss_parameter, help=bob.visioner.param.loss_parameter.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-o", "--optimization-type", metavar='OPT', type=str,
      dest='optimization_type', choices=bob.visioner.OPTIMIZATIONS, default=defp.optimization_type,
      help=bob.visioner.param.optimization_type.__doc__ + " (options: %s; default: %%(default)s)" % '|'.join(bob.visioner.OPTIMIZATIONS))
  parser.add_argument("-t", "--trainer-type", metavar='TRAINER', type=str,
      dest='training_model', choices=bob.visioner.TRAINERS, default=defp.training_model,
      help=bob.visioner.param.training_model.__doc__ + " (options: %s; default: %%(default)s)" % '|'.join(bob.visioner.TRAINERS))
  parser.add_argument("-m", "--max-rounds", metavar='INT', type=int,
      dest='max_rounds', default=defp.max_rounds, help=bob.visioner.param.max_rounds.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-n", "--num-of-bootstraps", metavar='INT', type=int,
      default=defp.num_of_bootstraps, dest='num_of_bootstraps',
      help=bob.visioner.param.num_of_bootstraps.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-T", "--training-data", metavar='FILE', type=str,
      default=defp.training_data, dest='training_data',
      help=bob.visioner.param.training_data.__doc__ + " (defaults to '%(default)s')")
  parser.add_argument("-U", "--num-of-training-samples", metavar='INT', 
      type=int, default=defp.num_of_train_samples, dest='num_of_train_samples',
      help=bob.visioner.param.num_of_train_samples.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-V", "--validation-data", metavar='FILE', type=str,
      default=defp.validation_data, dest='validation_data',
      help=bob.visioner.param.validation_data.__doc__ + " (defaults to '%(default)s')")
  parser.add_argument("-X", "--num-of-validation-samples", metavar='INT',
      dest='num_of_valid_samples', type=int, default=defp.num_of_valid_samples,
      help=bob.visioner.param.num_of_valid_samples.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-f", "--feature-type", metavar='FEATURE', type=str,
      choices=bob.visioner.MODELS, default=defp.feature_type,
      dest='feature_type', help=bob.visioner.param.feature_type.__doc__ + " (options: %s; default: %%(default)s)" % '|'.join(bob.visioner.MODELS))
  parser.add_argument("-S", "--feature-sharing", metavar='SHARING', type=str,
      choices=bob.visioner.SHARINGS, default=defp.feature_sharing,
      dest='feature_sharing', help=bob.visioner.param.feature_sharing.__doc__ + " (options: %s; default: %%(default)s)" % '|'.join(bob.visioner.SHARINGS))
  parser.add_argument("-p", "--feature-projections", metavar='INT', 
      type=int, default=defp.feature_projections, dest='feature_projections',
      help=bob.visioner.param.feature_projections.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-M", "--minimum-gt-overlap", metavar='FLOAT',
      type=float, default=defp.min_gt_overlap, dest='min_gt_overlap',
      help=bob.visioner.param.min_gt_overlap.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-W", "--sliding-windows", metavar='INT',
      type=int, default=defp.sliding_windows, dest='sliding_windows',
      help=bob.visioner.param.sliding_windows.__doc__ + " (defaults to %(default)s)")
  parser.add_argument("-Z", "--subwindow-labelling", metavar='TAGGER', type=str,
      choices=bob.visioner.TAGGERS, default=defp.subwindow_labelling,
      dest = 'subwindow_labelling', help=bob.visioner.param.subwindow_labelling.__doc__ + " (options: %s; default: %%(default)s)" % '|'.join(bob.visioner.TAGGERS))
  parser.add_argument("-y", "--threads", dest="threads", type=int,
      default=0, help="Set to zero to execute the training in the current thread, set to 1 or greater to spawn that many threads (defaults to %(default)s)")
  parser.add_argument("-v", "--verbose", dest="verbose",
      default=False, action='store_true',
      help="enable verbose output")
  parser.add_argument('model', metavar='FILE', type=str,
      help="File path where to store the model (in visioner format)")

  args = parser.parse_args()

  if args.threads < 0:
    parser.error("Number of threads should be greater or equal 0. The value '%d' is not valid" % args.threads)
    sys.exit(1)

  # now we read and set the parameters
  param = as_parameter(args)
  model = bob.visioner.Model(param)

  if args.verbose: print "Loading training and validation data..."
  start = time.clock()
  training = bob.visioner.Sampler(param, bob.visioner.SamplerType.Train,
      args.threads)
  validation = bob.visioner.Sampler(param, bob.visioner.SamplerType.Validation,
      args.threads)
  total = time.clock() - start
  if args.verbose: print "Ok. Loading time was %.2f seconds" % total

  if args.verbose: print "Training the model..."
  train_ok = model.train(training, validation)
  
  if not train_ok:
    raise RuntimeError, "A training error was detected... Cannot save model."

  total = time.clock() - start
  if args.verbose: print "Ok. Training time was %.2f seconds" % total

  if args.verbose: print "Saving the model at '%s'..." % args.model
  save_ok = model.save(args.model)
  if not save_ok:
    raise RuntimeError, "Could not save model."

  if args.verbose: print "Ok. Model saved." % total

  sys.exit(0)
