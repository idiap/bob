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

parser.add_option("-n",
                  "--noworld",
                  action="store_true",
                  dest="noworld",
                  default="False",
                  help="Do not use a world model")

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
  if not options.noworld:
    prior_gmm = torch.machine.GMMMachine(torch.io.HDF5File(options.world_model))
  gmm = torch.machine.GMMMachine(torch.io.HDF5File(options.model))

  # Compute the score
  scoreCL = 0.
  scoreWM = 0.
  for v in ar:
    scoreCL += gmm.forward(v.get())
    if not options.noworld:
      scoreWM += prior_gmm.forward(v.get())

  # scoreWM is equal to zero if noworld is not set
  score = (scoreCL - scoreWM) / n_blocks

  # Print the score
  print myfile + " " + str(score)

