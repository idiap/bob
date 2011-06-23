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

  gmm.weights = torch.core.array.load('data/gmm.init_weights.hdf5')
  gmm.means = torch.core.array.load('data/gmm.init_means.hdf5')
  gmm.variances = torch.core.array.load('data/gmm.init_variances.hdf5')
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

def NormalizeStdArrayset(path):
  arrayset = torch.io.Arrayset(path)
  arrayset.load()

  length = arrayset.shape[0]
  n_samples = len(arrayset)
  mean = torch.core.array.float64_1(length)
  std = torch.core.array.float64_1(length)

  mean.fill(0)
  std.fill(0)

  
  for i in range(0, n_samples):
    x = arrayset[i].get().cast('float64')
    mean += x
    std += (x ** 2)

  mean /= n_samples
  std /= n_samples
  std -= (mean ** 2)
  std = std ** 0.5 # sqrt(std)

  arStd = torch.io.Arrayset()
  for i in range(0, n_samples):
    x = arrayset[i].get().cast('float64')
    arStd.append(x / std)

  return (arStd,std)


class MyTrainer1(torch.trainer.KMeansTrainer):
  """Simple example of python trainer: """
  def __init__(self):
    torch.trainer.KMeansTrainer.__init__(self)
  
  def train(self, machine, data):
    a = torch.core.array.float64_2(2, 2)
    a[0, :] = data[1].get()
    a[1, :] = data[2].get()
    machine.means = a


class MyTrainer2(torch.trainer.overload.KMeansTrainer):
  """Simple example of python trainer: """
  def __init__(self):
    torch.trainer.overload.KMeansTrainer.__init__(self)
 
  def initialization(self, machine, data):
    print "Called by C++ method train()"
    torch.trainer.overload.KMeansTrainer.initialization(self, machine, data)
    print "Leaving initialization(), back into C++"


class TrainerTest(unittest.TestCase):
  """Performs various trainer tests."""
  
  def test00_kmeans(self):
    """Train a KMeansMachine"""

    # This files contains draws from two 1D Gaussian distributions:
    #   * 100 samples from N(-10,1)
    #   * 100 samples from N(10,1)
    data = torch.io.Arrayset("data/samplesFrom2G_f64.hdf5")

    machine = torch.machine.KMeansMachine(2, 1)

    trainer = torch.trainer.KMeansTrainer()
    trainer.train(machine, data)

    [variances, weights] = machine.getVariancesAndWeightsForEachCluster(data)
    m1 = machine.getMean(0)
    m2 = machine.getMean(1)

    # Check means [-10,10] / variances [1,1] / weights [0.5,0.5]
    if(m1<m2):
      means=torch.core.array.float64_1([m1[0],m2[0]],(2,))
    else:
      means=torch.core.array.float64_1([m2[0],m1[0]],(2,))
    self.assertTrue(equals(means, torch.core.array.float64_1([-10.,10.], (2,)), 2e-1))
    self.assertTrue(equals(variances,torch.core.array.float64_2([1.,1.], (2,1)), 2e-1))
    self.assertTrue(equals(weights,torch.core.array.float64_1([0.5,0.5], (2,)), 1e-3))

  def test01_kmeans(self):
    """Train a KMeansMachine"""

    (arStd,std) = NormalizeStdArrayset("data/faithful.torch3.hdf5")

    machine = torch.machine.KMeansMachine(2, 2)

    trainer = torch.trainer.KMeansTrainer()
    #trainer.seed = 1337
    trainer.train(machine, arStd)


    [variances, weights] = machine.getVariancesAndWeightsForEachCluster(arStd)
    means = machine.means

    multiplyVectorsByFactors(means, std)
    multiplyVectorsByFactors(variances, std ** 2)

    gmmWeights = torch.core.array.load('data/gmm.init_weights.hdf5')
    gmmMeans = torch.core.array.load('data/gmm.init_means.hdf5')
    gmmVariances = torch.core.array.load('data/gmm.init_variances.hdf5')

    if (means[0, 0] < means[1, 0]):
      means = flipRows(means)
      variances = flipRows(variances)
      weights = flipRows(weights)
    
    self.assertTrue(equals(means, gmmMeans, 1e-3))
    self.assertTrue(equals(weights, gmmWeights, 1e-3))
    self.assertTrue(equals(variances, gmmVariances, 1e-3))
    
  def test02_gmm_ML(self):
    """Train a GMMMachine with ML_GMMTrainer"""
    
    ar = torch.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    gmm = loadGMM()

    ml_gmmtrainer = torch.trainer.ML_GMMTrainer(True, True, True)
    ml_gmmtrainer.train(gmm, ar)

    #config = torch.io.HDF5File("data/gmm_ML.hdf5")
    #gmm.save(config)

    gmm_ref = torch.machine.GMMMachine(torch.io.HDF5File("data/gmm_ML.hdf5"))
    gmm_ref_32bit_release = torch.machine.GMMMachine(torch.io.HDF5File("data/gmm_ML_32bit_release.hdf5"))
    
    self.assertTrue((gmm == gmm_ref) or (gmm == gmm_ref_32bit_release))
    
  def test03_gmm_MAP(self):
    """Train a GMMMachine with MAP_GMMTrainer"""
    
    ar = torch.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    gmm = torch.machine.GMMMachine(torch.io.HDF5File("data/gmm_ML.hdf5"))
    gmmprior = torch.machine.GMMMachine(torch.io.HDF5File("data/gmm_ML.hdf5"))
    
    map_gmmtrainer = torch.trainer.MAP_GMMTrainer(16)
    map_gmmtrainer.setPriorGMM(gmmprior)
    map_gmmtrainer.train(gmm, ar)

    #config = torch.io.HDF5File("data/gmm_MAP.hdf5")
    #gmm.save(config)
    
    gmm_ref = torch.machine.GMMMachine(torch.io.HDF5File("data/gmm_MAP.hdf5"))
    gmm_ref_32bit_release = torch.machine.GMMMachine(torch.io.HDF5File("data/gmm_MAP_32bit_release.hdf5"))

    self.assertTrue((gmm == gmm_ref) or (gmm == gmm_ref_32bit_release))
    
  def test05_custom_trainer(self):
    """Custom python trainer"""
    
    ar = torch.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    mytrainer = MyTrainer1()

    machine = torch.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)
    
    for i in range(0, 2):
      print machine.means[i,:]
      self.assertTrue((ar[i+1].get() == machine.means[i, :]).all())


  def test06_custom_initialization(self):
    ar = torch.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    mytrainer = MyTrainer2()

    machine = torch.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)
    
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
