#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Tue May 10 11:35:58 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

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


class GMMTest(unittest.TestCase):
  """Performs various trainer tests."""
      
  def test01_gmm_ML(self):

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

  def test02_gmm_ML(self):

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
    
  def test03_gmm_MAP(self):

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
    
  def test04_gmm_MAP(self):

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
   
  def test05_gmm_MAP(self):
    
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

  def test06_gmm_test(self):

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
 
  def test07_custom_trainer(self):

    # Custom python trainer
    
    ar = bob.io.load(F("faithful.torch3_f64.hdf5"))
    
    mytrainer = MyTrainer1()

    machine = bob.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)
    
    for i in range(0, 2):
      self.assertTrue((ar[i+1] == machine.means[i, :]).all())

  def test08_custom_initialization(self):

    ar = bob.io.load(F("faithful.torch3_f64.hdf5"))
    
    mytrainer = MyTrainer2()

    machine = bob.machine.KMeansMachine(2, 2)
    mytrainer.train(machine, ar)

  def test09_overload_initialization(self):
    """Test introduces after ticket #87"""
    
    machine = bob.machine.KMeansMachine(2,1)
    data = numpy.array([[-3],[-2],[-1],[0.],[1.],[2.],[3.]])

    class MyKMeansTrainer(bob.trainer.overload.KMeansTrainer):
      """Simple example of python trainer: """
      def __init__(self):
        bob.trainer.overload.KMeansTrainer.__init__(self)
 
      def initialization(self, machine, data):
        bob.trainer.overload.KMeansTrainer.initialization(self, machine, data)
        machine.means = numpy.array([[-0.5], [ 0.5]])

    trainer = MyKMeansTrainer()
    trainer.convergence_threshold = 0.0005
    trainer.max_iterations = 1;
    trainer.train(machine, data) # After the initialization the means are still [0.,0.] (at the C++ level)
    self.assertFalse( numpy.isnan(machine.means).any())
