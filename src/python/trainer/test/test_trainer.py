#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <francois.moulin@idiap.ch>

"""Test trainer package
"""

import os, sys
import unittest
import torch
import random


def loadGMM():
  gmm = torch.machine.GMMMachine(2, 2)

  gmm.weights = torch.core.array.load('data/gmm.init_weights.bin')
  gmm.means = torch.core.array.load('data/gmm.init_means.bin')
  gmm.variances = torch.core.array.load('data/gmm.init_variances.bin')
  gmm.varianceThreshold = torch.core.array.array([0.001, 0.001], 'float64')

  return gmm


def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.rows()):
    for j in range(0, matrix.columns()):
      matrix[i, j] *= vector[j]

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()


class MyFrameSampler(torch.trainer.Sampler_FrameSample_):
  """Simple example of python sampler: get samples from an Arrayset"""
  def __init__(self):
    torch.trainer.Sampler_FrameSample_.__init__(self)
    self.arrayset = torch.database.Arrayset("data/faithful.torch3.bindata")
    self.arrayset.load()
  
  def getSample(self, index):
    return torch.machine.FrameSample(self.arrayset[index+1].get())
  
  def getNSamples(self):
    return len(self.arrayset)


class NormalizeStdFrameSampler(torch.trainer.Sampler_FrameSample_):
  """
  Sampler that opens an array set from a file and normalizes the standard deviation
  of the array set to 1
  """
  def __init__(self, path):
    torch.trainer.Sampler_FrameSample_.__init__(self)
    self.arrayset = torch.database.Arrayset(path)
    self.arrayset.load()
    
    length = self.arrayset.shape[0]
    n_samples = len(self.arrayset)
    mean = torch.core.array.float64_1(length)
    self.std = torch.core.array.float64_1(length)
    
    mean.fill(0)
    self.std.fill(0)
    
    for i in range(0, n_samples):
      x = self.arrayset[i+1].get().cast('float64')
      mean += x
      self.std += (x ** 2)
    
    mean /= n_samples
    self.std /= n_samples
    self.std -= (mean ** 2)
    self.std = self.std ** 0.5 # sqrt(std)
  
  
  def getSample(self, index):
    return torch.machine.FrameSample((self.arrayset[index+1].get().cast('float64') / self.std).cast('float32'))
  
  def getNSamples(self):
    return len(self.arrayset)


class MyTrainer(torch.trainer.Trainer_KMeansMachine_FrameSample_):
  """Simple example of python trainer: """
  def __init__(self):
    torch.trainer.Trainer_KMeansMachine_FrameSample_.__init__(self)
  
  def train(self, machine, data):
    a = torch.core.array.float64_2(2, 2)
    a[0, :] = data.getSample(0).getFrame().cast('float64')
    a[1, :] = data.getSample(1).getFrame().cast('float64')
    machine.means = a

class TerainerTest(unittest.TestCase):
  """Performs various trainer tests."""
  
  def test01_kmeans(self):
    """Train a KMeansMachine"""

    sampler = NormalizeStdFrameSampler("data/faithful.torch3.bindata")

    machine = torch.machine.KMeansMachine(2, 2)

    trainer = torch.trainer.KMeansTrainer()
    trainer.seed = 1337
    trainer.train(machine, sampler)


    [variances, weights] = machine.getVariancesAndWeightsForEachCluster(sampler)
    means = machine.means

    multiplyVectorsByFactors(means, sampler.std)
    multiplyVectorsByFactors(variances, sampler.std ** 2)

    gmmWeights = torch.core.array.load('data/gmm.init_weights.bin')
    gmmMeans = torch.core.array.load('data/gmm.init_means.bin')
    gmmVariances = torch.core.array.load('data/gmm.init_variances.bin')

    self.assertTrue(equals(means, gmmMeans, 1e-7))
    self.assertTrue(equals(weights, gmmWeights, 1e-7))
    self.assertTrue(equals(variances, gmmVariances, 1e-7))
    
  def test02_gmm_ML(self):
    """Train a GMMMachine with ML_GMMTrainer"""
    
    sampler = torch.trainer.SimpleFrameSampler(torch.database.Arrayset("data/faithful.torch3.bindata"))
    
    gmm = loadGMM()

    ml_gmmtrainer = torch.trainer.ML_GMMTrainer()

    print "GMM Before:"
    print gmm.print_()

    ml_gmmtrainer.train(gmm, sampler)
    
    print "GMM After:"
    print gmm.print_()

    #TODO Add asserts
    
  def test03_gmm_MAP(self):
    """Train a GMMMachine with MAP_GMMTrainer"""
    
    sampler = torch.trainer.SimpleFrameSampler(torch.database.Arrayset("data/faithful.torch3.bindata"))
    
    gmm = loadGMM()
    gmmprior = loadGMM()

    print gmm.print_()
    map_gmmtrainer = torch.trainer.MAP_GMMTrainer(16)
    map_gmmtrainer.setPriorGMM(gmmprior)
    map_gmmtrainer.train(gmm, sampler)
    print gmm.print_()
    
    #TODO Add asserts
    
  def test04_custom_samplers(self):
    """Custom python sampler"""
    
    sampler = torch.trainer.SimpleFrameSampler(torch.database.Arrayset("data/faithful.torch3.bindata"))
    mysampler = MyFrameSampler()

    self.assertTrue(sampler.getNSamples() == mysampler.getNSamples())

    for i in range(0, sampler.getNSamples()):
      self.assertTrue((sampler.getSample(i).getFrame() == mysampler.getSample(i).getFrame()).all())
    
  def test05_custom_trainer(self):
    """Custom python trainer"""
    
    sampler = torch.trainer.SimpleFrameSampler(torch.database.Arrayset("data/faithful.torch3.bindata"))
    
    mytrainer = MyTrainer()

    machine = torch.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, sampler)
    
    for i in range(0, 2):
      self.assertTrue((sampler.getSample(i).getFrame().cast('float64') == machine.means[i, :]).all())

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
