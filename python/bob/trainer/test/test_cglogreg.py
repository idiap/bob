#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Sat Sep 1 9:43:00 2012 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Test trainer package
"""

import os, sys
import unittest
import bob
import random
import numpy

class CGLogRegTest(unittest.TestCase):
  """Performs various tests for Linear Logistic Regression."""

  def test01_cglogreg(self):

    # Tests our LLR Trainer.
    positives = numpy.array([
      [1.,1.2,-1.],
      [2.,2.1,2.2],
      [3.,2.9,3.1],
      [4.,3.7,4.],
      [5.,5.2,4.9],
      [6.,6.1,5.9],
      [7.,7.,7.3],
      ], dtype='float64')

    negatives = numpy.array([
      [-10.,-9.2,-1.],
      [-5.,-4.1,-0.5],
      [-10.,-9.9,-1.8],
      [-5.,-5.4,-0.3],
      [-10.,-9.3,-0.7],
      [-5.,-4.5,-0.5],
      [-10.,-9.7,-1.2],
      [-5.,-4.8,-0.2],
      ], dtype='float64')

    # Expected trained machine
    #weights_ref= numpy.array([[13.5714], [19.3997], [-0.6432]])
    weights_ref= numpy.array([[1.75536], [2.69297], [-0.54142]])
    #bias_ref = numpy.array([55.3255])
    bias_ref = numpy.array([7.26999])

    # Features and expected outputs of the trained machine
    feat1 = numpy.array([1.,2.,3.])
    #out1 = 105.7668
    out1 = 12.78703
    feat2 = numpy.array([2.,3.,4.])
    #out2 = 138.0947
    out2 = 16.69394


    # Trains a machine (method 1)
    T = bob.trainer.CGLogRegTrainer(0.5, 1e-5, 30)
    machine1 = T.train(negatives,positives)

    # Makes sure results are good
    self.assertTrue( (abs(machine1.weights - weights_ref) < 2e-4).all() )
    self.assertTrue( (abs(machine1.biases - bias_ref) < 2e-4).all() )
    self.assertTrue( abs(machine1(feat1) - out1) < 2e-4 )
    self.assertTrue( abs(machine1(feat2) - out2) < 2e-4 )

    # Trains a machine (method 2)
    machine2 = bob.machine.LinearMachine()
    T.train(machine2, negatives, positives)

    # Makes sure results are good
    self.assertTrue( (abs(machine2.weights - weights_ref) < 2e-4).all() )
    self.assertTrue( (abs(machine2.biases - bias_ref) < 2e-4).all() )
    self.assertTrue( abs(machine2(feat1) - out1) < 2e-4 )
    self.assertTrue( abs(machine2(feat2) - out2) < 2e-4 )

    # Expected trained machine (with regularization)
    weights_ref= numpy.array([[0.54926], [0.58304], [0.06558]])
    bias_ref = numpy.array([0.27897])

    # Trains a machine (method 1)
    T = bob.trainer.CGLogRegTrainer(0.5, 1e-5, 30, 1.)
    machine1 = T.train(negatives, positives)

    # Makes sure results are good
    self.assertTrue( (abs(machine1.weights - weights_ref) < 2e-4).all() )
    self.assertTrue( (abs(machine1.biases - bias_ref) < 2e-4).all() )


  def test02_cglogreg_norm(self):
    # read some real test data;
    # for toy examples the results are quite different...

    pos1 = bob.io.load(bob.test.utils.datafile('positives_isv.hdf5', 'bob.trainer.test', 'data'))
    neg1 = bob.io.load(bob.test.utils.datafile('negatives_isv.hdf5', 'bob.trainer.test', 'data'))

    pos2 = bob.io.load(bob.test.utils.datafile('positives_lda.hdf5', 'bob.trainer.test', 'data'))
    neg2 = bob.io.load(bob.test.utils.datafile('negatives_lda.hdf5', 'bob.trainer.test', 'data'))

    negatives = numpy.vstack((neg1, neg2)).T
    positives = numpy.vstack((pos1, pos2)).T

    # Train the machine after mean-std norm
    T = bob.trainer.CGLogRegTrainer(0.5, 1e-10, 10000, mean_std_norm=True)
    machine = T.train(negatives,positives)

    # assert that mean and variance are correct
    mean = numpy.mean(numpy.vstack((positives, negatives)), 0)
    std = numpy.std(numpy.vstack((positives, negatives)), 0)

    self.assertTrue( (abs(machine.input_subtract - mean) < 1e-10).all() )
    self.assertTrue( (abs(machine.input_divide - std) < 1e-10).all() )

    # apply it to test data
    test1 = [1., -50.]
    test2 = [0.5, -86.]

    res1 = machine(test1)
    res2 = machine(test2)

    # normalize training data
    pos = numpy.vstack([(positives[i] - mean) / std for i in range(len(positives))])
    neg = numpy.vstack([(negatives[i] - mean) / std for i in range(len(negatives))])

    # re-train the machine; should give identical results
    T.mean_std_norm = False
    machine = T.train(neg, pos)
    machine.input_subtract = mean
    machine.input_divide = std

    # assert that the result is the same
    self.assertTrue( abs(machine(test1) - res1) < 1e-10 )
    self.assertTrue( abs(machine(test2) - res2) < 1e-10 )

    if not bob.core.is_debug():
      # try the training without normalization
      machine = T.train(negatives, positives)
      # check that the results are at least approximately equal
      # Note: lower values for epsilon and higher number of iterations improve the stability)
      self.assertTrue( abs(machine(test1) - res1) < 1e-3 )
      self.assertTrue( abs(machine(test2) - res2) < 1e-3 )


