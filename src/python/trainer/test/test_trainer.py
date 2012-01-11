#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <francois.moulin@idiap.ch>

"""Test trainer package
"""
import os, sys
import unittest
import bob
import random
import numpy

def loadGMM():
  gmm = bob.machine.GMMMachine(2, 2)

  gmm.weights = bob.io.load('data/gmm.init_weights.hdf5')
  gmm.means = bob.io.load('data/gmm.init_means.hdf5')
  gmm.variances = bob.io.load('data/gmm.init_variances.hdf5')
  gmm.varianceThreshold = numpy.array([0.001, 0.001], 'float64')

  return gmm

def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
      matrix[i, j] *= vector[j]

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

def flipRows(array):
  if len(array.shape) == 2:
    return numpy.array([numpy.array(array[1, :]), numpy.array(array[0, :])], 'float64')
  elif len(array.shape) == 1:
    return numpy.array([array[1], array[0]], 'float64')
  else:
    raise Exception('Input type not supportd by flipRows')

def NormalizeStdArrayset(path):
  arrayset = bob.io.Arrayset(path)
  arrayset.load()

  length = arrayset.shape[0]
  n_samples = len(arrayset)
  mean = numpy.ndarray(length, 'float64')
  std = numpy.ndarray(length, 'float64')

  mean.fill(0)
  std.fill(0)

  
  for i in range(0, n_samples):
    x = arrayset[i].astype('float64')
    mean += x
    std += (x ** 2)

  mean /= n_samples
  std /= n_samples
  std -= (mean ** 2)
  std = std ** 0.5 # sqrt(std)

  arStd = bob.io.Arrayset()
  for i in range(0, n_samples):
    x = arrayset[i].astype('float64')
    arStd.append(x / std)

  return (arStd,std)


class MyTrainer1(bob.trainer.KMeansTrainer):
  """Simple example of python trainer: """
  def __init__(self):
    bob.trainer.KMeansTrainer.__init__(self)
  
  def train(self, machine, data):
    a = numpy.ndarray((2, 2), 'float64')
    a[0, :] = data[1]
    a[1, :] = data[2]
    machine.means = a


class MyTrainer2(bob.trainer.overload.KMeansTrainer):
  """Simple example of python trainer: """
  def __init__(self):
    bob.trainer.overload.KMeansTrainer.__init__(self)
 
  def initialization(self, machine, data):
    print "Called by C++ method train()"
    bob.trainer.overload.KMeansTrainer.initialization(self, machine, data)
    print "Leaving initialization(), back into C++"


class TrainerTest(unittest.TestCase):
  """Performs various trainer tests."""
  
  def test00_kmeans(self):
    """Train a KMeansMachine"""

    # This files contains draws from two 1D Gaussian distributions:
    #   * 100 samples from N(-10,1)
    #   * 100 samples from N(10,1)
    data = bob.io.Arrayset("data/samplesFrom2G_f64.hdf5")

    machine = bob.machine.KMeansMachine(2, 1)

    trainer = bob.trainer.KMeansTrainer()
    trainer.train(machine, data)

    [variances, weights] = machine.getVariancesAndWeightsForEachCluster(data)
    m1 = machine.getMean(0)
    m2 = machine.getMean(1)

    # Check means [-10,10] / variances [1,1] / weights [0.5,0.5]
    if(m1<m2): means=numpy.array(([m1[0],m2[0]]), 'float64')
    else: means=numpy.array(([m2[0],m1[0]]), 'float64')
    self.assertTrue(equals(means, numpy.array([-10.,10.]), 2e-1))
    self.assertTrue(equals(variances, numpy.array([1.,1.]), 2e-1))
    self.assertTrue(equals(weights, numpy.array([0.5,0.5]), 1e-3))

  def test01_kmeans(self):
    """Train a KMeansMachine"""

    (arStd,std) = NormalizeStdArrayset("data/faithful.torch3.hdf5")

    machine = bob.machine.KMeansMachine(2, 2)

    trainer = bob.trainer.KMeansTrainer()
    #trainer.seed = 1337
    trainer.train(machine, arStd)


    [variances, weights] = machine.getVariancesAndWeightsForEachCluster(arStd)
    means = machine.means

    multiplyVectorsByFactors(means, std)
    multiplyVectorsByFactors(variances, std ** 2)

    gmmWeights = bob.io.load('data/gmm.init_weights.hdf5')
    gmmMeans = bob.io.load('data/gmm.init_means.hdf5')
    gmmVariances = bob.io.load('data/gmm.init_variances.hdf5')

    if (means[0, 0] < means[1, 0]):
      means = flipRows(means)
      variances = flipRows(variances)
      weights = flipRows(weights)
    
    self.assertTrue(equals(means, gmmMeans, 1e-3))
    self.assertTrue(equals(weights, gmmWeights, 1e-3))
    self.assertTrue(equals(variances, gmmVariances, 1e-3))
    
  def test02_gmm_ML(self):
    """Train a GMMMachine with ML_GMMTrainer"""
    
    ar = bob.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    gmm = loadGMM()

    ml_gmmtrainer = bob.trainer.ML_GMMTrainer(True, True, True)
    ml_gmmtrainer.train(gmm, ar)

    #config = bob.io.HDF5File("data/gmm_ML.hdf5")
    #gmm.save(config)

    gmm_ref = bob.machine.GMMMachine(bob.io.HDF5File("data/gmm_ML.hdf5"))
    gmm_ref_32bit_debug = bob.machine.GMMMachine(bob.io.HDF5File("data/gmm_ML_32bit_debug.hdf5"))
    gmm_ref_32bit_release = bob.machine.GMMMachine(bob.io.HDF5File("data/gmm_ML_32bit_release.hdf5"))
    
    self.assertTrue((gmm == gmm_ref) or (gmm == gmm_ref_32bit_release) or (gmm == gmm_ref_32bit_debug))

  def test03_gmm_ML(self):
    """Train a GMMMachine with ML_GMMTrainer and compare to torch3vision reference"""
   
    ar = bob.io.Arrayset("data/dataNormalized.hdf5") 

    # Initialize GMMMachine
    gmm = bob.machine.GMMMachine(5, 45)
    gmm.means = bob.io.load("data/meansAfterKMeans.hdf5").astype('float64')
    gmm.variances = bob.io.load("data/variancesAfterKMeans.hdf5").astype('float64')
    gmm.weights = numpy.exp(bob.io.load("data/weightsAfterKMeans.hdf5").astype('float64'))
   
    threshold = 0.001
    gmm.setVarianceThresholds(threshold)
    
    # Initialize ML Trainer
    prior = 0.001
    max_iter_gmm = 25+1
    accuracy = 0.00001
    ml_gmmtrainer = bob.trainer.ML_GMMTrainer(True, True, True, prior)
    ml_gmmtrainer.maxIterations = max_iter_gmm
    ml_gmmtrainer.convergenceThreshold = accuracy

    # Run ML
    ml_gmmtrainer.train(gmm, ar)

    # Test results
    # Load torch3vision reference
    meansML_ref = bob.io.load("data/meansAfterML.hdf5")
    variancesML_ref = bob.io.load("data/variancesAfterML.hdf5")
    weightsML_ref = bob.io.load("data/weightsAfterML.hdf5")

    # Compare to current results
    self.assertTrue(equals(gmm.means, meansML_ref, 3e-3))
    self.assertTrue(equals(gmm.variances, variancesML_ref, 3e-3))
    self.assertTrue(equals(gmm.weights, weightsML_ref, 1e-4))
    
  def test04_gmm_MAP(self):
    """Train a GMMMachine with MAP_GMMTrainer"""
    
    ar = bob.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    gmm = bob.machine.GMMMachine(bob.io.HDF5File("data/gmm_ML.hdf5"))
    gmmprior = bob.machine.GMMMachine(bob.io.HDF5File("data/gmm_ML.hdf5"))
    
    map_gmmtrainer = bob.trainer.MAP_GMMTrainer(16)
    map_gmmtrainer.setPriorGMM(gmmprior)
    map_gmmtrainer.train(gmm, ar)

    #config = bob.io.HDF5File("data/gmm_MAP.hdf5")
    #gmm.save(config)
    
    gmm_ref = bob.machine.GMMMachine(bob.io.HDF5File("data/gmm_MAP.hdf5"))
    gmm_ref_32bit_release = bob.machine.GMMMachine(bob.io.HDF5File("data/gmm_MAP_32bit_release.hdf5"))

    self.assertTrue((gmm == gmm_ref) or (gmm == gmm_ref_32bit_release))
    
  def test05_gmm_MAP(self):
    """Train a GMMMachine with MAP_GMMTrainer and compare with matlab reference"""

    map_adapt = bob.trainer.MAP_GMMTrainer(4., True, False, False, 0.)
    data = bob.io.load('../../machine/test/data/data.hdf5')
    means = bob.io.load('../../machine/test/data/means.hdf5')
    variances = bob.io.load('../../machine/test/data/variances.hdf5')
    weights = bob.io.load('../../machine/test/data/weights.hdf5')

    gmm = bob.machine.GMMMachine(2,50)
    gmm.means = means
    gmm.variances = variances
    gmm.weights = weights

    arrayset = bob.io.Arrayset()
    arrayset.append(data)

    map_adapt.setPriorGMM(gmm)

    gmm_adapted = bob.machine.GMMMachine(2,50)
    gmm_adapted.means = means
    gmm_adapted.variances = variances
    gmm_adapted.weights = weights

    map_adapt.maxIterations = 1
    map_adapt.train(gmm_adapted,arrayset)

    new_means = bob.io.load('data/new_adapted_mean.hdf5')

    # Compare to matlab reference
    self.assertTrue(equals(new_means[0,:], gmm_adapted.means[:,0], 1e-4))
    self.assertTrue(equals(new_means[1,:], gmm_adapted.means[:,1], 1e-4))
   
  def test06_gmm_MAP(self):
    """Train a GMMMachine with MAP_GMMTrainer and compare to torch3vision reference"""
   
    ar = bob.io.Arrayset("data/dataforMAP.hdf5") 

    # Initialize GMMMachine
    n_gaussians = 5
    n_inputs = 45
    prior_gmm = bob.machine.GMMMachine(n_gaussians, n_inputs)
    prior_gmm.means = bob.io.load("data/meansAfterML.hdf5")
    prior_gmm.variances = bob.io.load("data/variancesAfterML.hdf5")
    prior_gmm.weights = bob.io.load("data/weightsAfterML.hdf5")
  
    threshold = 0.001
    prior_gmm.setVarianceThresholds(threshold)
    
    # Initialize MAP Trainer
    relevance_factor = 0.1
    prior = 0.001
    max_iter_gmm = 1
    accuracy = 0.00001
    map_factor = 0.5
    map_gmmtrainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, False, False, prior)
    map_gmmtrainer.maxIterations = max_iter_gmm
    map_gmmtrainer.convergenceThreshold = accuracy
    map_gmmtrainer.setPriorGMM(prior_gmm) 
    map_gmmtrainer.setT3MAP(map_factor); 

    gmm = bob.machine.GMMMachine(n_gaussians, n_inputs)
    gmm.setVarianceThresholds(threshold)

    # Train
    map_gmmtrainer.train(gmm, ar)
 
    # Test results
    # Load torch3vision reference
    meansMAP_ref = bob.io.load("data/meansAfterMAP.hdf5")
    variancesMAP_ref = bob.io.load("data/variancesAfterMAP.hdf5")
    weightsMAP_ref = bob.io.load("data/weightsAfterMAP.hdf5")

    # Compare to current results
    # Gaps are quite large. This might be explained by the fact that there is no 
    # adaptation of a given Gaussian in torch3 when the corresponding responsibilities
    # are below the responsibilities threshold
    self.assertTrue(equals(gmm.means, meansMAP_ref, 2e-1))
    self.assertTrue(equals(gmm.variances, variancesMAP_ref, 1e-4))
    self.assertTrue(equals(gmm.weights, weightsMAP_ref, 1e-4))

  def test07_gmm_test(self):
    """Test a GMMMachine by computing scores against a model and compare to 
    torch3vision reference"""
   
    ar = bob.io.Arrayset("data/dataforMAP.hdf5") 

    # Initialize GMMMachine
    n_gaussians = 5
    n_inputs = 45
    gmm = bob.machine.GMMMachine(n_gaussians, n_inputs)
    gmm.means = bob.io.load("data/meansAfterML.hdf5")
    gmm.variances = bob.io.load("data/variancesAfterML.hdf5")
    gmm.weights = bob.io.load("data/weightsAfterML.hdf5")
  
    threshold = 0.001
    gmm.setVarianceThresholds(threshold)
    
    # Test against the model
    score_mean_ref = -1.50379e+06
    score = 0.
    for v in ar: score += gmm.forward(v)
    score /= len(ar)
  
    # Compare current results to torch3vision
    self.assertTrue(abs(score-score_mean_ref)/score_mean_ref<1e-4)
 
  def test08_custom_trainer(self):
    """Custom python trainer"""
    
    ar = bob.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    mytrainer = MyTrainer1()

    machine = bob.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)
    
    for i in range(0, 2):
      print machine.means[i,:]
      self.assertTrue((ar[i+1] == machine.means[i, :]).all())

  def test09_custom_initialization(self):
    ar = bob.io.Arrayset("data/faithful.torch3_f64.hdf5")
    
    mytrainer = MyTrainer2()

    machine = bob.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)
    
if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()
