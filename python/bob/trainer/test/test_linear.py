#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Jun 10 16:43:41 2011 +0200
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

"""Test trainers for the LinearMachine
"""

import os, sys
import unittest
import bob
import random
import numpy

class LinearTest(unittest.TestCase):
  """Performs various trainer tests for the LinearMachine."""
  
  def test01a_pca_via_svd(self):

    # Tests our SVD/PCA extractor.
    data = numpy.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2., 1.6],
        [1., 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
        ], dtype='float64')

    # Expected results
    eig_val_correct = numpy.array([1.28402771, 0.0490834], 'float64')
    eig_vec_correct = numpy.array([[-0.6778734, -0.73517866], [-0.73517866, 0.6778734]], 'float64')

    T = bob.trainer.SVDPCATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    self.assertTrue( (abs(machine.weights - eig_vec_correct) < 1e-6).all() )
    self.assertTrue( (abs(eig_vals - eig_val_correct) < 1e-6).all() )

  def test01b_pca_via_svd(self):

    # Tests our SVD/PCA extractor.
    data = numpy.array([
      [1,2, 3,5,7],
      [2,4,19,0,2],
      [3,6, 5,3,3],
      [4,8,13,4,2],
      ], dtype='float64')

    # Expected results
    eig_val_correct = numpy.array([61.9870996, 9.49613738, 1.85009634, 0.],
        'float64')

    T = bob.trainer.SVDPCATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    self.assertTrue( (abs(eig_vals - eig_val_correct) < 1e-6).all() )
    self.assertTrue( machine.weights.shape[0] == 5 and machine.weights.shape[1] == 4 )

  def test02a_fisher_lda(self):

    # Tests our Fisher/LDA trainer for linear machines for a simple 2-class
    # "fake" problem:
    data = [
        numpy.array([
          [2.5, 2.4],
          [2.2, 2.9],
          [1.9, 2.2],
          [3.1, 3.0],
          [2.3, 2.7],
          ], dtype='float64'),
        numpy.array([
          [0.5, 0.7],
          [2., 1.6],
          [1., 1.1],
          [1.5, 1.6],
          [1.1, 0.9],
          ], dtype='float64'),
        ]

    # Expected results
    exp_trans_data = [
        [1.0019, 3.1205, 0.9405, 2.4962, 2.2949], 
        [-2.9042, -1.3179, -2.0172, -0.7720, -2.8428]
        ]
    exp_mean = numpy.array([1.8100, 1.9100])
    exp_val = numpy.array([24.27536526])
    exp_mach = numpy.array([[-0.291529], [0.956562]])

    T = bob.trainer.FisherLDATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    self.assertTrue( (abs(machine.input_subtract - exp_mean) < 1e-6).all() )
    self.assertTrue( (abs(machine.weights - exp_mach) < 1e-6).all() )
    self.assertTrue( (abs(eig_vals - exp_val) < 1e-6).all() )

  def test02b_fisher_lda_bis(self):

    # Tests our Fisher/LDA trainer for linear machines for a simple 2-class
    # "fake" problem:
    data = [
        numpy.array([
          [2.5, 2.4, 2.5],
          [2.2, 2.9, 3.],
          [1.9, 2.2, 2.],
          [3.1, 3.0, 3.1],
          [2.3, 2.7, 2.4],
          ], dtype='float64'),
        numpy.array([
          [-0.5, -0.7, -1.],
          [-2., -1.6, -2.],
          [-1., -1.1, -1.],
          [-1.5, -1.6, -1.6],
          [-1.1, -0.9, -1.],
          ], dtype='float64'),
        ]

    # Expected results
    exp_mean = numpy.array([0.59, 0.73, 0.64])
    exp_val = numpy.array([1.52746000e+02, 1.91783967e-13])
    exp_mach = numpy.array([[0.14322439, 0.03851495], [-0.98379062, 0.69764105], [0.10790173, -0.71541147]])

    T = bob.trainer.FisherLDATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    machine.resize(3,1) # eigenvalue close to 0 are not significant (just keep the first one)
    self.assertTrue( (abs(machine.input_subtract - exp_mean) < 1e-6).all() )
    self.assertTrue( (abs(eig_vals[0:1] - exp_val[0:1]) < 1e-6).all() )
    self.assertTrue( (abs(machine.weights[:,0] - exp_mach[:,0]) < 1e-6).all() )

  def test03_ppca(self):

    # Tests our Probabilistic PCA trainer for linear machines for a simple 
    # problem:
    ar=numpy.array([
      [1, 2, 3],
      [2, 4, 19],
      [3, 6, 5],
      [4, 8, 13],
      ], dtype='float64')
    
    # Expected llh 1 and 2 (Reference values)
    exp_llh1 =  -32.8443
    exp_llh2 =  -30.8559
   
    # Do two iterations of EM to check the training procedure 
    T = bob.trainer.EMPCATrainer(2)
    m = bob.machine.LinearMachine()
    # Initialization of the trainer
    T.initialization(m, ar)
    # Sets ('random') initialization values for test purposes
    w_init = numpy.array([1.62945, 0.270954, 1.81158, 1.67002, 0.253974,
      1.93774], 'float64').reshape(3,2)
    sigma2_init = 1.82675
    m.weights = w_init
    T.sigma2 = sigma2_init
    # Checks that the log likehood matches the reference one
    # This should be sufficient to check everything as it requires to use
    # the new value of W and sigma2 
    # This does an E-Step, M-Step, computes the likelihood, and compares it to
    # the reference value obtained using matlab
    T.e_step(m, ar)
    T.m_step(m, ar)
    llh1 = T.compute_likelihood(m)
    self.assertTrue( abs(exp_llh1 - llh1) < 2e-4)
    T.e_step(m, ar)
    T.m_step(m, ar)
    llh2 = T.compute_likelihood(m)
    self.assertTrue( abs(exp_llh2 - llh2) < 2e-4)


  def test04_whitening_initialization(self):

    # Constructors and comparison operators
    t1 = bob.trainer.WhiteningTrainer()
    t2 = bob.trainer.WhiteningTrainer()
    t3 = bob.trainer.WhiteningTrainer(t2)
    t4 = t3
    self.assertTrue( t1 == t2)
    self.assertFalse( t1 != t2)
    self.assertTrue( t1.is_similar_to(t2) )
    self.assertTrue( t1 == t3)
    self.assertFalse( t1 != t3)
    self.assertTrue( t1.is_similar_to(t3) )
    self.assertTrue( t1 == t4)
    self.assertFalse( t1 != t4)
    self.assertTrue( t1.is_similar_to(t4) )


  def test05_whitening_train(self):

    # Tests our Whitening extractor.
    data = numpy.array([[ 1.2622, -1.6443, 0.1889],
                        [ 0.4286, -0.8922, 1.3020],
                        [-0.6613,  0.0430, 0.6377],
                        [-0.8718, -0.4788, 0.3988],
                        [-0.0098, -0.3121,-0.1807],
                        [ 0.4301,  0.4886, -0.1456]])
    sample = numpy.array([1, 2, 3.])

    # Expected results (from matlab)
    mean_ref = numpy.array([0.096324163333333, -0.465965438333333, 0.366839091666667])
    whit_ref = numpy.array([[1.608410253685985,                  0,                  0],
                            [1.079813355720326,  1.411083365535711,                  0],
                            [0.693459921529905,  0.571417184139332,  1.800117179839927]])
    sample_whitened_ref = numpy.array([5.942255453628436, 4.984316201643742, 4.739998188373740])
    
    # Runs whitening (first method)
    t = bob.trainer.WhiteningTrainer()
    m = bob.machine.LinearMachine(3,3)
    t.train(m, data)
    s = m.forward(sample)

    # Makes sure results are good
    eps = 1e-4
    self.assertTrue( numpy.allclose(m.input_subtract, mean_ref, eps, eps) )
    self.assertTrue( numpy.allclose(m.weights, whit_ref, eps, eps) )
    self.assertTrue( numpy.allclose(s, sample_whitened_ref, eps, eps) )

    # Runs whitening (second method)
    m2 = t.train(data)
    s2 = m2.forward(sample)

    # Makes sure results are good
    eps = 1e-4
    self.assertTrue( numpy.allclose(m2.input_subtract, mean_ref, eps, eps) )
    self.assertTrue( numpy.allclose(m2.weights, whit_ref, eps, eps) )
    self.assertTrue( numpy.allclose(s2, sample_whitened_ref, eps, eps) )

