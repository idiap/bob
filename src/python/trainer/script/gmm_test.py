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
  ar = torch.io.Arrayset(myfile)

  # Compute the number of blocks
  n_blocks = len(ar)

  # Load the gmm
  prior_gmm = torch.machine.GMMMachine(torch.io.HDF5File(options.world_model))
  gmm = torch.machine.GMMMachine(torch.io.HDF5File(options.model))

  # Compute the score
  scoreCL = 0.
  scoreWM = 0.
  for id in range(0, n_blocks):
    scoreCL += gmm.forward(ar[id].get())
    scoreWM += prior_gmm.forward(ar[id].get())

  score = (scoreCL - scoreWM) / n_blocks

  # Print the score
  print myfile + " " + str(score)

