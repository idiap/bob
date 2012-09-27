#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Tue May 10 11:35:58 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test trainer package
"""
import os, sys
import unittest
import bob
import random
import numpy
import pkg_resources

def F(f, module=None):
  """Returns the test file on the "data" subdirectory"""
  if module is None:
    return pkg_resources.resource_filename(__name__, os.path.join('data', f))
  return pkg_resources.resource_filename('bob.%s.test' % module, 
      os.path.join('data', f))

def loadGMM():
  gmm = bob.machine.GMMMachine(2, 2)

  gmm.weights = bob.io.load(F('gmm.init_weights.hdf5'))
  gmm.means = bob.io.load(F('gmm.init_means.hdf5'))
  gmm.variances = bob.io.load(F('gmm.init_variances.hdf5'))
  gmm.variance_threshold = numpy.array([0.001, 0.001], 'float64')

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

def NormalizeStdArray(path):

  array = bob.io.load(path).astype('float64')
  std = array.std(axis=0)
  return (array/std, std)

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

    # Trains a KMeansMachine

    # This files contains draws from two 1D Gaussian distributions:
    #   * 100 samples from N(-10,1)
    #   * 100 samples from N(10,1)
    data = bob.io.load(F("samplesFrom2G_f64.hdf5"))

    machine = bob.machine.KMeansMachine(2, 1)

    trainer = bob.trainer.KMeansTrainer()
    trainer.train(machine, data)

    [variances, weights] = machine.get_variances_and_weights_for_each_cluster(data)
    m1 = machine.get_mean(0)
    m2 = machine.get_mean(1)

    # Check means [-10,10] / variances [1,1] / weights [0.5,0.5]
    if(m1<m2): means=numpy.array(([m1[0],m2[0]]), 'float64')
    else: means=numpy.array(([m2[0],m1[0]]), 'float64')
    self.assertTrue(equals(means, numpy.array([-10.,10.]), 2e-1))
    self.assertTrue(equals(variances, numpy.array([1.,1.]), 2e-1))
    self.assertTrue(equals(weights, numpy.array([0.5,0.5]), 1e-3))

  def test01_kmeans(self):

    # Trains a KMeansMachine

    (arStd,std) = NormalizeStdArray(F("faithful.torch3.hdf5"))

    machine = bob.machine.KMeansMachine(2, 2)

    trainer = bob.trainer.KMeansTrainer()
    #trainer.seed = 1337
    trainer.train(machine, arStd)

    [variances, weights] = machine.get_variances_and_weights_for_each_cluster(arStd)
    means = machine.means

    multiplyVectorsByFactors(means, std)
    multiplyVectorsByFactors(variances, std ** 2)

    gmmWeights = bob.io.load(F('gmm.init_weights.hdf5'))
    gmmMeans = bob.io.load(F('gmm.init_means.hdf5'))
    gmmVariances = bob.io.load(F('gmm.init_variances.hdf5'))

    if (means[0, 0] < means[1, 0]):
      means = flipRows(means)
      variances = flipRows(variances)
      weights = flipRows(weights)
   
    self.assertTrue(equals(means, gmmMeans, 1e-3))
    self.assertTrue(equals(weights, gmmWeights, 1e-3))
    self.assertTrue(equals(variances, gmmVariances, 1e-3))
    
  def test02_gmm_ML(self):

    # Trains a GMMMachine with ML_GMMTrainer
    
    ar = bob.io.load(F("faithful.torch3_f64.hdf5"))
    
    gmm = loadGMM()

    ml_gmmtrainer = bob.trainer.ML_GMMTrainer(True, True, True)
    ml_gmmtrainer.train(gmm, ar)

    #config = bob.io.HDF5File(F('gmm_ML.hdf5", 'w'))
    #gmm.save(config)

    gmm_ref = bob.machine.GMMMachine(bob.io.HDF5File(F('gmm_ML.hdf5')))
    gmm_ref_32bit_debug = bob.machine.GMMMachine(bob.io.HDF5File(F('gmm_ML_32bit_debug.hdf5')))
    gmm_ref_32bit_release = bob.machine.GMMMachine(bob.io.HDF5File(F('gmm_ML_32bit_release.hdf5')))

    self.assertTrue((gmm == gmm_ref) or (gmm == gmm_ref_32bit_release) or (gmm == gmm_ref_32bit_debug))

  def test03_gmm_ML(self):

    # Trains a GMMMachine with ML_GMMTrainer; compares to an old reference
   
    ar = bob.io.load(F('dataNormalized.hdf5')) 

    # Initialize GMMMachine
    gmm = bob.machine.GMMMachine(5, 45)
    gmm.means = bob.io.load(F('meansAfterKMeans.hdf5')).astype('float64')
    gmm.variances = bob.io.load(F('variancesAfterKMeans.hdf5')).astype('float64')
    gmm.weights = numpy.exp(bob.io.load(F('weightsAfterKMeans.hdf5')).astype('float64'))
   
    threshold = 0.001
    gmm.set_variance_thresholds(threshold)
    
    # Initialize ML Trainer
    prior = 0.001
    max_iter_gmm = 25
    accuracy = 0.00001
    ml_gmmtrainer = bob.trainer.ML_GMMTrainer(True, True, True, prior)
    ml_gmmtrainer.max_iterations = max_iter_gmm
    ml_gmmtrainer.convergence_threshold = accuracy

    # Run ML
    ml_gmmtrainer.train(gmm, ar)

    # Test results
    # Load torch3vision reference
    meansML_ref = bob.io.load(F('meansAfterML.hdf5'))
    variancesML_ref = bob.io.load(F('variancesAfterML.hdf5'))
    weightsML_ref = bob.io.load(F('weightsAfterML.hdf5'))

    # Compare to current results
    self.assertTrue(equals(gmm.means, meansML_ref, 3e-3))
    self.assertTrue(equals(gmm.variances, variancesML_ref, 3e-3))
    self.assertTrue(equals(gmm.weights, weightsML_ref, 1e-4))
    
  def test04_gmm_MAP(self):

    # Train a GMMMachine with MAP_GMMTrainer
    
    ar = bob.io.load(F('faithful.torch3_f64.hdf5'))
    
    gmm = bob.machine.GMMMachine(bob.io.HDF5File(F("gmm_ML.hdf5")))
    gmmprior = bob.machine.GMMMachine(bob.io.HDF5File(F("gmm_ML.hdf5")))
    
    map_gmmtrainer = bob.trainer.MAP_GMMTrainer(16)
    map_gmmtrainer.set_prior_gmm(gmmprior)
    map_gmmtrainer.train(gmm, ar)

    #config = bob.io.HDF5File(F('gmm_MAP.hdf5", 'w'))
    #gmm.save(config)
    
    gmm_ref = bob.machine.GMMMachine(bob.io.HDF5File(F('gmm_MAP.hdf5')))
    #gmm_ref_32bit_release = bob.machine.GMMMachine(bob.io.HDF5File(F('gmm_MAP_32bit_release.hdf5')))

    self.assertTrue((equals(gmm.means,gmm_ref.means,1e-3) and equals(gmm.variances,gmm_ref.variances,1e-3) and equals(gmm.weights,gmm_ref.weights,1e-3)))
    
  def test05_gmm_MAP(self):

    # Train a GMMMachine with MAP_GMMTrainer and compare with matlab reference

    map_adapt = bob.trainer.MAP_GMMTrainer(4., True, False, False, 0.)
    data = bob.io.load(F('data.hdf5', 'machine'))
    data = data.reshape((1, data.shape[0])) # make a 2D array out of it
    means = bob.io.load(F('means.hdf5', 'machine'))
    variances = bob.io.load(F('variances.hdf5', 'machine'))
    weights = bob.io.load(F('weights.hdf5', 'machine'))

    gmm = bob.machine.GMMMachine(2,50)
    gmm.means = means
    gmm.variances = variances
    gmm.weights = weights

    map_adapt.set_prior_gmm(gmm)

    gmm_adapted = bob.machine.GMMMachine(2,50)
    gmm_adapted.means = means
    gmm_adapted.variances = variances
    gmm_adapted.weights = weights

    map_adapt.max_iterations = 1
    print data.shape
    map_adapt.train(gmm_adapted, data)

    new_means = bob.io.load(F('new_adapted_mean.hdf5'))

    # Compare to matlab reference
    self.assertTrue(equals(new_means[0,:], gmm_adapted.means[:,0], 1e-4))
    self.assertTrue(equals(new_means[1,:], gmm_adapted.means[:,1], 1e-4))
   
  def test06_gmm_MAP(self):
    
    # Train a GMMMachine with MAP_GMMTrainer; compares to old reference
   
    ar = bob.io.load(F('dataforMAP.hdf5')) 

    # Initialize GMMMachine
    n_gaussians = 5
    n_inputs = 45
    prior_gmm = bob.machine.GMMMachine(n_gaussians, n_inputs)
    prior_gmm.means = bob.io.load(F('meansAfterML.hdf5'))
    prior_gmm.variances = bob.io.load(F('variancesAfterML.hdf5'))
    prior_gmm.weights = bob.io.load(F('weightsAfterML.hdf5'))
  
    threshold = 0.001
    prior_gmm.set_variance_thresholds(threshold)
    
    # Initialize MAP Trainer
    relevance_factor = 0.1
    prior = 0.001
    max_iter_gmm = 1
    accuracy = 0.00001
    map_factor = 0.5
    map_gmmtrainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, False, False, prior)
    map_gmmtrainer.max_iterations = max_iter_gmm
    map_gmmtrainer.convergence_threshold = accuracy
    map_gmmtrainer.set_prior_gmm(prior_gmm) 
    map_gmmtrainer.set_t3_map(map_factor); 

    gmm = bob.machine.GMMMachine(n_gaussians, n_inputs)
    gmm.set_variance_thresholds(threshold)

    # Train
    map_gmmtrainer.train(gmm, ar)
 
    # Test results
    # Load torch3vision reference
    meansMAP_ref = bob.io.load(F('meansAfterMAP.hdf5'))
    variancesMAP_ref = bob.io.load(F('variancesAfterMAP.hdf5'))
    weightsMAP_ref = bob.io.load(F('weightsAfterMAP.hdf5'))

    # Compare to current results
    # Gaps are quite large. This might be explained by the fact that there is no 
    # adaptation of a given Gaussian in torch3 when the corresponding responsibilities
    # are below the responsibilities threshold
    self.assertTrue(equals(gmm.means, meansMAP_ref, 2e-1))
    self.assertTrue(equals(gmm.variances, variancesMAP_ref, 1e-4))
    self.assertTrue(equals(gmm.weights, weightsMAP_ref, 1e-4))

  def test07_gmm_test(self):

    # Tests a GMMMachine by computing scores against a model and compare to 
    # an old reference
   
    ar = bob.io.load(F('dataforMAP.hdf5')) 

    # Initialize GMMMachine
    n_gaussians = 5
    n_inputs = 45
    gmm = bob.machine.GMMMachine(n_gaussians, n_inputs)
    gmm.means = bob.io.load(F('meansAfterML.hdf5'))
    gmm.variances = bob.io.load(F('variancesAfterML.hdf5'))
    gmm.weights = bob.io.load(F('weightsAfterML.hdf5'))
  
    threshold = 0.001
    gmm.set_variance_thresholds(threshold)
    
    # Test against the model
    score_mean_ref = -1.50379e+06
    score = 0.
    for v in ar: score += gmm.forward(v)
    score /= len(ar)
  
    # Compare current results to torch3vision
    self.assertTrue(abs(score-score_mean_ref)/score_mean_ref<1e-4)
 
  def test08_custom_trainer(self):

    # Custom python trainer
    
    ar = bob.io.load(F("faithful.torch3_f64.hdf5"))
    
    mytrainer = MyTrainer1()

    machine = bob.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)
    
    for i in range(0, 2):
      self.assertTrue((ar[i+1] == machine.means[i, :]).all())

  def test09_custom_initialization(self):

    ar = bob.io.load(F("faithful.torch3_f64.hdf5"))
    
    mytrainer = MyTrainer2()

    machine = bob.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)
