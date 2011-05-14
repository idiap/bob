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

def flipRows(array):
  if type(array).__name__ == 'float64_2':
    return torch.core.array.array([array[1, :].as_ndarray(), array[0, :].as_ndarray()], 'float64')
  elif type(array).__name__ == 'float64_1':
    return torch.core.array.array([array[1], array[0]], 'float64')
  else:
    raise Exception('Input type not supportd by flipRows')

class MyFrameSampler(torch.trainer.Sampler_FrameSample_):
  """Simple example of python sampler: get samples from an Arrayset"""
  def __init__(self,path):
    torch.trainer.Sampler_FrameSample_.__init__(self)
    self.arrayset = torch.database.Arrayset(path)
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

class TrainerTest(unittest.TestCase):
  """Performs various trainer tests."""
  
  def test00_kmeans(self):
    """Train a KMeansMachine"""

    # This files contains draws from two 1D Gaussian distributions:
    #   * 100 samples from N(-10,1)
    #   * 100 samples from N(10,1)
    sampler = MyFrameSampler("data/samplesFrom2G.hdf5")

    machine = torch.machine.KMeansMachine(2, 1)

    trainer = torch.trainer.KMeansTrainer()
    trainer.train(machine, sampler)

    [variances, weights] = machine.getVariancesAndWeightsForEachCluster(sampler)
    m1 = machine.getMean(0)
    m2 = machine.getMean(1)

    # Check means [-10,10] / variances [1,1] / weights [0.5,0.5]
    if(m1<m2):
      means=torch.core.array.float64_1([m1[0],m2[0]],(2,))
    else:
      means=torch.core.array.float64_1([m2[0],m1[0]],(2,))
    self.assertTrue(equals(means, torch.core.array.float64_1([-10.,10.], (2,)), 2e-1))
    self.assertTrue(equals(variances,torch.core.array.float64_2([1.,1.],(2,1)),2e-1))
    self.assertTrue(equals(weights,torch.core.array.float64_1([0.5,0.5],(2,)),1e-3))

  def test01_kmeans(self):
    """Train a KMeansMachine"""

    sampler = NormalizeStdFrameSampler("data/faithful.torch3.bindata")

    machine = torch.machine.KMeansMachine(2, 2)

    trainer = torch.trainer.KMeansTrainer()
    #trainer.seed = 1337
    trainer.train(machine, sampler)


    [variances, weights] = machine.getVariancesAndWeightsForEachCluster(sampler)
    means = machine.means

    multiplyVectorsByFactors(means, sampler.std)
    multiplyVectorsByFactors(variances, sampler.std ** 2)

    gmmWeights = torch.core.array.load('data/gmm.init_weights.bin')
    gmmMeans = torch.core.array.load('data/gmm.init_means.bin')
    gmmVariances = torch.core.array.load('data/gmm.init_variances.bin')

    if (means[0, 0] < means[1, 0]):
      means = flipRows(means)
      variances = flipRows(variances)
      weights = flipRows(weights)
    
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
    mysampler = MyFrameSampler("data/faithful.torch3.bindata")

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

  def test06_pca(self):
    # Define input samples
    s = range(0,10)
    s[0] = torch.core.array.float32_1([2.5, 2.4],(2,))
    s[1] = torch.core.array.float32_1([0.5, 0.7],(2,))
    s[2] = torch.core.array.float32_1([2.2, 2.9],(2,))
    s[3] = torch.core.array.float32_1([1.9, 2.2],(2,))
    s[4] = torch.core.array.float32_1([3.1, 3.0],(2,))
    s[5] = torch.core.array.float32_1([2.3, 2.7],(2,))
    s[6] = torch.core.array.float32_1([2., 1.6],(2,))
    s[7] = torch.core.array.float32_1([1., 1.1],(2,))
    s[8] = torch.core.array.float32_1([1.5, 1.6],(2,))
    s[9] = torch.core.array.float32_1([1.1, 0.9],(2,))
    a = torch.database.Arrayset()
    for i in range(0,10):
      a.append(s[i])
    sampler = torch.trainer.SimpleFrameSampler(a)

    mymachine = torch.machine.EigenMachine()
    mytrainer = torch.trainer.SVDPCATrainer()
    mytrainer.train(mymachine, sampler)

    print mymachine.getEigenvalues()
    print mymachine.getEigenvectors()

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
