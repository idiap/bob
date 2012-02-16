#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Fri May 27 15:47:40 2011 +0200
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

import bob
import os, sys
import optparse
import math
import numpy

def NormalizeStdArrayset(arrayset):
  arrayset.load()

  length = arrayset.shape[0]
  n_samples = len(arrayset)
  mean = numpy.ndarray(length)
  std = numpy.ndarray(length)

  mean.fill(0)
  std.fill(0)

  for array in arrayset:
    x = array.astype('float64')
    mean += x
    std += (x ** 2)

  mean /= n_samples
  std /= n_samples
  std -= (mean ** 2)
  std = std ** 0.5 # sqrt(std)

  arStd = bob.io.Arrayset()
  for array in arrayset:
    arStd.append(array.astype('float64') / std)

  return (arStd,std)


def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
      matrix[i, j] *= vector[j]


import fileinput
from optparse import OptionParser

def main():
  usage = "usage: %prog [options] <input_files> "

  parser = OptionParser(usage)
  parser.set_description("Train a GMM model")

  parser.add_option("-o",
                    "--output-file",
                    dest="output_file",
                    help="Output file",
                    type="string",
                    default="wm.hdf5")
  parser.add_option("-g",
                    "--n-gaussians",
                    dest="n_gaussians",
                    help="",
                    type="int",
                    default=5)
  parser.add_option("--iterk",
                    dest="iterk",
                    help="Max number of iterations of KMeans",
                    type="int",
                    default=25)
  parser.add_option("--iterg",
                    dest="iterg",
                    help="Max number of iterations of GMM",
                    type="int",
                    default=25)
  parser.add_option("-e",
                    dest="convergence_threshold",
                    help="End accuracy",
                    type="float",
                    default=1e-05)
  parser.add_option("-v",
                    "--variance-threshold",
                    dest="variance_threshold",
                    help="Variance threshold",
                    type="float",
                    default=0.001)
  parser.add_option('--no-update-weights',
                    action="store_true",
                    dest="no_update_weights",
                    help="Do not update the weights",
                    default=False)
  parser.add_option('--no-update-means',
                    action="store_true",
                    dest="no_update_means",
                    help="Do not update the means",
                    default=False)
  parser.add_option('--no-adapt-variances',
                    action="store_true",
                    dest="no_update_variances",
                    help="Do not update the variances",
                    default=False)
  parser.add_option("-n",
                    "--no-norm",
                    action="store_true",
                    dest="no_norm",
                    help="Do not normalize input features for KMeans",
                    default=False)
  parser.add_option('--self-test',
                    action="store_true",
                    dest="test",
                    help=optparse.SUPPRESS_HELP,
                    default=False)

  (options, args) = parser.parse_args()

  if options.test:
    if os.path.exists("/tmp/input.hdf5"):
      os.remove("/tmp/input.hdf5")
    
    options.output_file = "/tmp/wm.hdf5"
    arrayset = bob.io.Arrayset()
    array1 = numpy.array([ 0,  1,  2,  3], 'float64')
    arrayset.append(array1)
    array2 = numpy.array([ 3,  1,  5,  2], 'float64')
    arrayset.append(array2)
    array3 = numpy.array([ 6,  7,  2,  5], 'float64')
    arrayset.append(array3)
    array4 = numpy.array([ 3,  6,  2,  3], 'float64')
    arrayset.append(array4)
    array5 = numpy.array([ 9,  8,  6,  4], 'float64')
    arrayset.append(array5)

    options.n_gaussians = 1
    arrayset.save("/tmp/input.hdf5")

    f = open("/tmp/input.lst", 'w')
    f.write("/tmp/input.hdf5\n")
    f.close()
    
    args.append("/tmp/input.lst")


  # Read the file list
  filelist = []
  for line in fileinput.input(args):
    filelist.append(line.rstrip('\r\n'))

  # Create an arrayset from the input files
  ar = bob.io.Arrayset()
  for myfile in filelist:
    myarrayset = bob.io.Arrayset(myfile)
    n_blocks = len(myarrayset)
    for b in range(0,n_blocks): ar.append(myarrayset[b])

  # Compute input size
  input_size = ar.shape[0]

  # Create a normalized sampler
  if options.no_norm:
    normalizedAr = ar
  else:
    (normalizedAr,stdAr) = NormalizeStdArrayset(ar)
    
  # Create the machines
  kmeans = bob.machine.KMeansMachine(options.n_gaussians, input_size)
  gmm = bob.machine.GMMMachine(options.n_gaussians, input_size)

  # Create the KMeansTrainer
  kmeansTrainer = bob.trainer.KMeansTrainer()
  kmeansTrainer.convergenceThreshold = options.convergence_threshold
  kmeansTrainer.maxIterations = options.iterk

  # Train the KMeansTrainer
  kmeansTrainer.train(kmeans, normalizedAr)

  [variances, weights] = kmeans.getVariancesAndWeightsForEachCluster(normalizedAr)
  means = kmeans.means

  # Undo normalization
  if not options.no_norm:
    multiplyVectorsByFactors(means, stdAr)
    multiplyVectorsByFactors(variances, stdAr ** 2)

  # Initialize gmm
  gmm.means = means
  gmm.variances = variances
  gmm.weights = weights
  gmm.setVarianceThresholds(options.variance_threshold)

  # Train gmm
  trainer = bob.trainer.ML_GMMTrainer(not options.no_update_means, not options.no_update_variances, not options.no_update_weights)
  trainer.convergenceThreshold = options.convergence_threshold
  trainer.maxIterations = options.iterg
  trainer.train(gmm, ar)

  # Save gmm
  config = bob.io.HDF5File(options.output_file)
  gmm.save(config)

  if options.test:
    os.remove("/tmp/input.hdf5")
    os.remove("/tmp/input.lst")
    
    if not os.path.exists("/tmp/wm.hdf5"):
      sys.exit(1)
    else:
      os.remove("/tmp/wm.hdf5")
