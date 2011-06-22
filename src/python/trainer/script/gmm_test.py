#!/usr/bin/env python

import torch
import os, sys
import optparse
import math


#class FileListFrameSampler(torch.trainer.Sampler_FrameSample_):
"""Get samples from a list of files Arrayset"""
"""
  def __init__(self, list_files, n_blocks):
    torch.trainer.Sampler_FrameSample_.__init__(self)
    self.list_files = list_files

    if n_blocks == None:
      # If the number of blocks is not provided, get it from the first file
      self.n_blocks = torch.io.Array(self.list_files[0]).get().extent(0)
    else:
      self.n_blocks = n_blocks
    
    self.last_index_file = -1
    
  
  def getSample(self, index):
    # Compute the file index for index
    index_file = int(math.floor(index / self.n_blocks))
    # Compute the array index for index
    index_array = index - (index_file * self.n_blocks)
    
    # We don't reload the input file if it is the same as the last one
    if self.last_index_file != index_file:
      self.arrays = torch.io.Array(self.list_files[index_file]).get()
      self.last_index_file = index_file

    # Get the right row
    array = self.arrays[index_array, :]
    
    if type(array).__name__ == 'float64_t':
      return torch.machine.FrameSample(array)
    else:
      return torch.machine.FrameSample(array.cast('float64'))
  
  def getNSamples(self):
    return len(self.list_files)*self.n_blocks
"""


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

