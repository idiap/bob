#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri Jan 18 12:46:00 2013 +0200
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

"""Test K-Means algorithm
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

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

def kmeans_plus_plus(machine, data, seed):
  """Python implementation of K-Means++ (initialization)"""
  n_data = data.shape[0]
  mt = bob.core.random.mt19937(seed)
  rng = bob.core.random.uniform_int32(0, n_data-1)
  index = rng(mt)
  machine.set_mean(0, data[index,:])
  weights = numpy.zeros(shape=(n_data,), dtype=numpy.float64)

  for m in range(1,machine.dim_c):
    for s in range(n_data):
      s_cur = data[s,:]
      w_cur = machine.get_distance_from_mean(s_cur, 0)
      for i in range(m):
        w_cur = min(machine.get_distance_from_mean(s_cur, i), w_cur)
      weights[s] = w_cur
    weights *= weights
    weights /= numpy.sum(weights)
    rng_d = bob.core.random.discrete_int32(weights)
    index = rng_d(mt)
    machine.set_mean(m, data[index,:])


def NormalizeStdArray(path):
  array = bob.io.load(path).astype('float64')
  std = array.std(axis=0)
  return (array/std, std)

def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
      matrix[i, j] *= vector[j]

def flipRows(array):
  if len(array.shape) == 2:
    return numpy.array([numpy.array(array[1, :]), numpy.array(array[0, :])], 'float64')
  elif len(array.shape) == 1:
    return numpy.array([array[1], array[0]], 'float64')
  else:
    raise Exception('Input type not supportd by flipRows')

class KMeansTest(unittest.TestCase):
  """Performs various trainer tests."""
  
  if hasattr(bob.trainer.KMeansTrainer, 'KMEANS_PLUS_PLUS'):
    def test00_kmeans_plus_plus(self):

      # Tests the K-Means++ initialization
      dim_c = 5
      dim_d = 7
      n_samples = 150
      data = numpy.random.randn(n_samples,dim_d)
      seed = 0
      
      # C++ implementation
      machine = bob.machine.KMeansMachine(dim_c, dim_d)
      trainer = bob.trainer.KMeansTrainer()
      trainer.seed = seed
      trainer.initialization_method = bob.trainer.KMeansTrainer.KMEANS_PLUS_PLUS
      trainer.initialization(machine, data)

      # Python implementation
      py_machine = bob.machine.KMeansMachine(dim_c, dim_d)
      kmeans_plus_plus(py_machine, data, seed)
      self.assertTrue(equals(machine.means, py_machine.means, 1e-8))

  def test01_kmeans_noduplicate(self):
    # Data/dimensions
    dim_c = 2
    dim_d = 3
    seed = 0
    data = numpy.array([[1,2,3],[1,2,3],[1,2,3],[4,5,6.]])
    # Defines machine and trainer
    machine = bob.machine.KMeansMachine(dim_c, dim_d)
    trainer = bob.trainer.KMeansTrainer()
    trainer.seed = seed
    trainer.initialization_method = bob.trainer.KMeansTrainer.RANDOM_NO_DUPLICATE
    trainer.initialization(machine, data)
    # Makes sure that the two initial mean vectors selected are different
    self.assertFalse(equals(machine.get_mean(0), machine.get_mean(1), 1e-8))
 
  def test02_kmeans_a(self):

    # Trains a KMeansMachine
    # This files contains draws from two 1D Gaussian distributions:
    #   * 100 samples from N(-10,1)
    #   * 100 samples from N(10,1)
    data = bob.io.load(F("samplesFrom2G_f64.hdf5"))

    machine = bob.machine.KMeansMachine(2, 1)

    trainer = bob.trainer.KMeansTrainer()
    trainer.train(machine, data)

    [variances, weights] = machine.get_variances_and_weights_for_each_cluster(data)
    variances_b = numpy.ndarray(shape=(2,1), dtype=numpy.float64)
    weights_b = numpy.ndarray(shape=(2,), dtype=numpy.float64)
    machine.__get_variances_and_weights_for_each_cluster_init__(variances_b, weights_b)
    machine.__get_variances_and_weights_for_each_cluster_acc__(data, variances_b, weights_b)
    machine.__get_variances_and_weights_for_each_cluster_fin__(variances_b, weights_b)
    m1 = machine.get_mean(0)
    m2 = machine.get_mean(1)

    # Check means [-10,10] / variances [1,1] / weights [0.5,0.5]
    if(m1<m2): means=numpy.array(([m1[0],m2[0]]), 'float64')
    else: means=numpy.array(([m2[0],m1[0]]), 'float64')
    self.assertTrue(equals(means, numpy.array([-10.,10.]), 2e-1))
    self.assertTrue(equals(variances, numpy.array([1.,1.]), 2e-1))
    self.assertTrue(equals(weights, numpy.array([0.5,0.5]), 1e-3))

    self.assertTrue(equals(variances, variances_b, 1e-8))
    self.assertTrue(equals(weights, weights_b, 1e-8))

  def test03_kmeans_b(self):

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

    # Check comparison operators
    trainer1 = bob.trainer.KMeansTrainer()
    trainer2 = bob.trainer.KMeansTrainer()
    trainer1.rng = trainer2.rng
    self.assertTrue( trainer1 == trainer2)
    self.assertFalse( trainer1 != trainer2)
    trainer1.max_iterations = 1337
    self.assertFalse( trainer1 == trainer2)
    self.assertTrue( trainer1 != trainer2)

    # Check that there is no duplicate means during initialization
    machine = bob.machine.KMeansMachine(2, 1)
    trainer = bob.trainer.KMeansTrainer()
    trainer.initialization_method = bob.trainer.KMeansTrainer.RANDOM_NO_DUPLICATE
    data = numpy.array([[1.], [1.], [1.], [1.], [1.], [1.], [2.], [3.]])
    trainer.train(machine, data)
    self.assertFalse( numpy.isnan(machine.means).any())

