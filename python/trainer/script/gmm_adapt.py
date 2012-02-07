#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Fri May 27 15:47:40 2011 +0200
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

import bob
import os, sys
import optparse
import math


import fileinput
from optparse import OptionParser

usage = "usage: %prog [options] <input_files> "

parser = OptionParser(usage)
parser.set_description("Adapt a GMM model")

parser.add_option("-o",
                  "--output-file",
                  dest="output_file",
                  help="Output file",
                  type="string",
                  default="train.hdf5")
parser.add_option("-p",
                  "--prior-model",
                  dest="prior_model",
                  help="Prior model",
                  type="string",
                  default="wm.hdf5")
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
parser.add_option('--adapt-weight',
                  action="store_true",
                  dest="adapt_weight",
                  help="Adapt weight",
                  default=False)
parser.add_option('--adapt-variance',
                  action="store_true",
                  dest="adapt_variance",
                  help="Adapt variance",
                  default=False)
parser.add_option("-r",
                  "--relevance-factor",
                  dest="relevance_factor",
                  help="Relevance factor",
                  type="float",
                  default=0.001)
parser.add_option("--responsibilities-threshold",
                  dest="responsibilities_threshold",
                  help="Mean and variance update responsibilities threshold",
                  type="float",
                  default=0.)
parser.add_option("--set-bob3-map",
                  action="store_true",
                  dest="bob3_map",
                  help="Use bob3-like MAP adaptation rather than Reynolds'one",
                  default=False)
parser.add_option("--alpha-bob3-map",
                  dest="alpha_bob3",
                  help="Set alpha to use with bob3-like MAP adaptation",
                  type="float",
                  default=0.5)


(options, args) = parser.parse_args()

# Read the list of file
filelist = []
for line in fileinput.input(args):
  filelist.append(line.rstrip('\r\n'))

# Create a sampler for the input files
ar = bob.io.Arrayset()
for myfile in filelist:
  myarrayset = bob.io.Arrayset(myfile)
  n_blocks = len(myarrayset)
  for b in range(0,n_blocks):
    x = myarrayset[b].get()
    ar.append(x)

# Load prior gmm
prior_gmm = bob.machine.GMMMachine(bob.io.HDF5File(options.prior_model))
prior_gmm.setVarianceThresholds(options.variance_threshold)

# Create trainer
if options.responsibilities_threshold == 0.:
  trainer = bob.trainer.MAP_GMMTrainer(options.relevance_factor, True, options.adapt_variance, options.adapt_weight)
else:
  trainer = bob.trainer.MAP_GMMTrainer(options.relevance_factor, True, options.adapt_variance, options.adapt_weight, options.responsibilities_threshold)
trainer.convergenceThreshold = options.convergence_threshold
trainer.maxIterations = options.iterg
trainer.setPriorGMM(prior_gmm)

if options.bob3_map:
  trainer.setT3MAP(options.alpha_bob3)

# Load gmm
gmm = bob.machine.GMMMachine(bob.io.HDF5File(options.prior_model))
gmm.setVarianceThresholds(options.variance_threshold)

# Train gmm
trainer.train(gmm, ar)

# Save gmm
gmm.save(bob.io.HDF5File(options.output_file))
