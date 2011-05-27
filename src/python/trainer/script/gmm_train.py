#!/usr/bin/env python

import torch
import os, sys
import optparse
import math


class FileListFrameSampler(torch.trainer.Sampler_FrameSample_):
  """Get samples from a list of files Arrayset"""
  def __init__(self, list_files, n_blocks):
    torch.trainer.Sampler_FrameSample_.__init__(self)
    self.list_files = list_files

    if n_blocks == None:
      # If the number of blocks is not provided, get it from the first file
      self.n_blocks = torch.database.Array(self.list_files[0]).get().extent(0)
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
      self.arrays = torch.database.Array(self.list_files[index_file]).get()
      self.last_index_file = index_file

    # Get the right row
    array = self.arrays[index_array, :]
    
    if type(array).__name__ == 'float64_t':
      return torch.machine.FrameSample(array)
    else:
      return torch.machine.FrameSample(array.cast('float64'))
  
  def getNSamples(self):
    return len(self.list_files)*self.n_blocks

class NormalizeStdFrameSampler(torch.trainer.Sampler_FrameSample_):
  """
  Sampler that opens an array set from a file and normalizes the standard deviation
  of the array set to 1
  """
  def __init__(self, sampler):
    torch.trainer.Sampler_FrameSample_.__init__(self)
    self.sampler = sampler

    length = sampler.getSample(0).getFrame().extent(0)
    n_samples = sampler.getNSamples()
    mean = torch.core.array.float64_1(length)
    self.std = torch.core.array.float64_1(length)

    mean.fill(0)
    self.std.fill(0)

    for i in range(0, n_samples):
      x = sampler.getSample(i).getFrame()
      mean += x
      self.std += (x ** 2)

    mean /= n_samples
    self.std /= n_samples
    self.std -= (mean ** 2)
    self.std = self.std ** 0.5 # sqrt(std)


  def getSample(self, index):
    return torch.machine.FrameSample(self.sampler.getSample(index).getFrame() / self.std)

  def getNSamples(self):
    return self.sampler.getNSamples()

def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.rows()):
    for j in range(0, matrix.columns()):
      matrix[i, j] *= vector[j]

import fileinput
from optparse import OptionParser

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
  array = torch.core.array.array([[ 0,  1,  2,  3],
                                  [ 3,  1,  5,  2],
                                  [ 6,  7,  2,  5],
                                  [ 3,  6,  2,  3],
                                  [ 9,  8,  6,  4]
                                  ],
                                  'float64')

  options.n_gaussians = 1
  torch.database.Array(array).save("/tmp/input.hdf5")

  f = open("/tmp/input.lst", 'w')
  f.write("/tmp/input.hdf5\n")
  f.close()
  
  args.append("/tmp/input.lst")


# Read the file list
filelist = []
for line in fileinput.input(args):
  filelist.append(line.rstrip('\r\n'))

# Create a sampler for the input files
sampler = FileListFrameSampler(filelist, None)

# Compute input size
input_size = sampler.getSample(0).getFrame().extent(0)

# Create a normalized sampler
normalizedSampler = NormalizeStdFrameSampler(sampler)

# Create the machines
kmeans = torch.machine.KMeansMachine(options.n_gaussians, input_size)
gmm = torch.machine.GMMMachine(options.n_gaussians, input_size)

# Create the KMeansTrainer
kmeansTrainer = torch.trainer.KMeansTrainer()
kmeansTrainer.convergenceThreshold = options.convergence_threshold
kmeansTrainer.maxIterations = options.iterk

# Train the KMeansTrainer
kmeansTrainer.train(kmeans, normalizedSampler)

[variances, weights] = kmeans.getVariancesAndWeightsForEachCluster(normalizedSampler)
means = kmeans.means

multiplyVectorsByFactors(means, normalizedSampler.std)
multiplyVectorsByFactors(variances, normalizedSampler.std ** 2)

# Initialize gmm
gmm.means = means
gmm.variances = variances
gmm.weights = weights
gmm.setVarianceThreshold = options.variance_threshold

# Train gmm
trainer = torch.trainer.ML_GMMTrainer(True, True, True)
trainer.convergenceThreshold = options.convergence_threshold
trainer.maxIterations = options.iterg
trainer.train(gmm, sampler)

# Save gmm
config = torch.config.Configuration()
gmm.save(config)
config.save(options.output_file)

if options.test:
  os.remove("/tmp/input.hdf5")
  os.remove("/tmp/input.lst")
  
  if not os.path.exists("/tmp/wm.hdf5"):
    sys.exit(1)
  else:
    os.remove("/tmp/wm.hdf5")
