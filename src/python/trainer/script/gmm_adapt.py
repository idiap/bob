#!/usr/bin/env python

import torch
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
parser.add_option("-r",
                  "--relevance-factor",
                  dest="relevance_factor",
                  help="Relevance factor",
                  type="float",
                  default=0.001)

(options, args) = parser.parse_args()

# Read the list of file
filelist = []
for line in fileinput.input(args):
  filelist.append(line.rstrip('\r\n'))

# Create a sampler for the input files
ar = torch.database.Arrayset()
for myfile in filelist:
  myarray = torch.database.Array(myfile)
  n_blocks = myarray.shape[0]
  for b in range(0,n_blocks):
    x = myarray.get().cast('float64')[b,:]
    ar.append(x)

# Compute input size
input_size = ar.shape[0]

# Load prior gmm
prior_gmm = torch.machine.GMMMachine(torch.config.Configuration(options.prior_model))
prior_gmm.setVarianceThreshold = options.variance_threshold

# Create trainer
trainer = torch.trainer.MAP_GMMTrainer(options.relevance_factor)
trainer.convergenceThreshold = options.convergence_threshold
trainer.maxIterations = options.iterg
trainer.setPriorGMM(prior_gmm)

# Load gmm
gmm = torch.machine.GMMMachine(torch.config.Configuration(options.prior_model))
gmm.setVarianceThreshold = options.variance_threshold

# Train gmm
trainer.train(gmm, ar)

# Save gmm
config = torch.config.Configuration()
gmm.save(config)
config.save(options.output_file)
