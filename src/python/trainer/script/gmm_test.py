#!/usr/bin/env python

import torch
import os, sys
import optparse
import math

import fileinput
from optparse import OptionParser

usage = "usage: %prog [options] <input_files> "

parser = OptionParser(usage)
parser.set_description("Test a GMM model")

parser.add_option("-m",
                  "--model",
                  dest="model",
                  help="Client model",
                  type="string",
                  default="train.hdf5")
parser.add_option("-w",
                  "--world-model",
                  dest="world_model",
                  help="World model",
                  type="string",
                  default="wm.hdf5")

(options, args) = parser.parse_args()

filelist = []
for line in fileinput.input(args):
  myfile = line.rstrip('\r\n')

  # Create data with only one file
  ar = torch.io.Arrayset()
  myarray = torch.io.Array(myfile)
  n_blocks = myarray.shape[0]
  for b in range(0,n_blocks):
    x = myarray.get().cast('float64')[b,:]
    ar.append(x)

  # Compute input size
  input_size = ar.shape[0]

  # Load the gmm
  prior_gmm = torch.machine.GMMMachine(torch.config.Configuration(options.world_model))
  gmm = torch.machine.GMMMachine(torch.config.Configuration(options.model))

  # Compute the score
  scoreCL = 0.
  scoreWM = 0.
  ids = ar.ids()
  for v, id in ids:
    scoreCL += gmm.forward(ar[id].get())
    scoreWM += prior_gmm.forward(ar[id].get())

  score = scoreCL - scoreWM

  # Print the score
  print filelist[0] + " " + str(score)

