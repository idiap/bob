#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Jul 14 18:53:07 2011 +0200
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

"""Tests for RProp MLP training.
"""

import os, sys
import unittest
import bob
import numpy

class PythonRProp:
  """A simplified (and slower) version of RProp training written in python.
  
  This version of the code is probably easier to understand than the C++
  version. Both algorithms should, essentially, be the same, except for the
  performance for obvious reasons.
  """

  def __init__(self, train_biases=True):
    # Our state
    self.DW = None #delta matrixes for weights
    self.DB = None #delta matrixes for biases
    self.PDW = None #partial derivatives for weights
    self.PDB = None #partial derivatives for biases
    self.PPDW = None #previous partial derivatives for weights
    self.PPDB = None #previous partial derivatives for biases
    self.train_biases = train_biases

  def reset(self):
    """Resets our internal state"""

    self.DW = None #delta matrixes for weights
    self.DB = None #delta matrixes for biases
    self.PDW = None #partial derivatives for weights
    self.PDB = None #partial derivatives for biases
    self.PPDW = None #previous partial derivatives for weights
    self.PPDB = None #previous partial derivatives for biases

  def train(self, machine, input, target):

    def sign(x):
      """A handy sign function"""
      if (x == 0): return 0
      if (x < 0) : return -1
      return +1

    def logistic(x):
      return 1 / (1 + numpy.exp(-x))

    def logistic_bwd(x):
      return x * (1-x)

    def tanh(x):
      return numpy.tanh(x)

    def tanh_bwd(x):
      return (1 - x**2)

    def linear(x):
      return x

    def linear_bwd(x):
      return 1
    
    # some constants for RProp
    DELTA0 = 0.1
    DELTA_MIN = 1e-6
    DELTA_MAX = 50
    ETA_MINUS = 0.5
    ETA_PLUS = 1.2

    W = machine.weights #weights
    B = machine.biases #biases

    if machine.activation == bob.machine.Activation.TANH:
      forward = tanh
      backward = tanh_bwd
    elif machine.activation == bob.machine.Activation.LOG:
      forward = logistic
      backward = logistic_bwd
    elif machine.activation == bob.machine.Activation.LINEAR:
      forward = linear
      backward = linear_bwd
    else:
      raise RuntimeError, "Cannot deal with activation %s" % machine.activation
    
    #simulated bias input...
    BI = [numpy.zeros((input.shape[0],), 'float64') for k in B]
    for k in BI: k.fill(1)

    #state
    if self.DW is None: #first run or just after a reset()
      self.DW = [numpy.empty_like(k) for k in W]
      for k in self.DW: k.fill(DELTA0)
      self.DB = [numpy.empty_like(k) for k in B]
      for k in self.DB: k.fill(DELTA0)
      self.PPDW = [numpy.empty_like(k) for k in W]
      for k in self.PPDW: k.fill(0)
      self.PPDB = [numpy.empty_like(k) for k in B]
      for k in self.PPDB: k.fill(0)

    # Instantiate partial outputs and errors
    O = [None for k in B]
    O.insert(0, input) # an extra slot for the input
    E = [None for k in B]

    # Feeds forward
    for k in range(len(W)):
      O[k+1] = numpy.dot(O[k], W[k])
      for sample in range(O[k+1].shape[0]):
        O[k+1][sample,:] += B[k]
      O[k+1] = forward(O[k+1])

    # Feeds backward
    E[-1] = backward(O[-1]) * (O[-1] - target) #last layer
    for k in reversed(range(len(W)-1)): #for all remaining layers
      E[k] = backward(O[k+1]) * numpy.dot(E[k+1], W[k+1].transpose(1,0))

    # Calculates partial derivatives, accumulate
    self.PDW = [numpy.dot(O[k].transpose(1,0), E[k]) for k in range(len(W))]
    self.PDB = [numpy.dot(BI[k], E[k]) for k in range(len(W))]

    # Updates weights and biases
    WUP = [i * j for (i,j) in zip(self.PPDW, self.PDW)]
    BUP = [i * j for (i,j) in zip(self.PPDB, self.PDB)]

    # Iterate over each weight and bias and see what to do:
    for k, up in enumerate(WUP):
      for i in range(up.shape[0]):
        for j in range(up.shape[1]):
          if up[i,j] > 0:
            self.DW[k][i,j] = min(self.DW[k][i,j]*ETA_PLUS, DELTA_MAX)
            W[k][i,j] -= sign(self.PDW[k][i,j]) * self.DW[k][i,j]
            self.PPDW[k][i,j] = self.PDW[k][i,j]
          elif up[i,j] < 0:
            self.DW[k][i,j] = max(self.DW[k][i,j]*ETA_MINUS, DELTA_MIN)
            self.PPDW[k][i,j] = 0
          elif up[i,j] == 0:
            W[k][i,j] -= sign(self.PDW[k][i,j]) * self.DW[k][i,j]
            self.PPDW[k][i,j] = self.PDW[k][i,j]
    machine.weights = W

    if self.train_biases:
      for k, up in enumerate(BUP):
        for i in range(up.shape[0]):
          if up[i] > 0:
            self.DB[k][i] = min(self.DB[k][i]*ETA_PLUS, DELTA_MAX)
            B[k][i] -= sign(self.PDB[k][i]) * self.DB[k][i]
            self.PPDB[k][i] = self.PDB[k][i]
          elif up[i] < 0:
            self.DB[k][i] = max(self.DB[k][i]*ETA_MINUS, DELTA_MIN)
            self.PPDB[k][i] = 0
          elif up[i] == 0:
            B[k][i] -= sign(self.PDB[k][i]) * self.DB[k][i]
            self.PPDB[k][i] = self.PDB[k][i]
      machine.biases = B

    else:
      machine.biases = 0

class RPropTest(unittest.TestCase):
  """Performs various RProp MLP training tests."""

  def test01_Initialization(self):

    # Initializes an MLPRPropTrainer and checks all seems consistent
    # with the proposed API.
    machine = bob.machine.MLP((4, 1))
    machine.activation = bob.machine.Activation.LINEAR
    B = 10
    trainer = bob.trainer.MLPRPropTrainer(machine, B)
    self.assertEqual( trainer.batchSize, B )
    self.assertTrue ( trainer.isCompatible(machine) )
    self.assertTrue ( trainer.trainBiases )

    machine = bob.machine.MLP((7, 2))
    self.assertFalse ( trainer.isCompatible(machine) )

    trainer.trainBiases = False
    self.assertFalse ( trainer.trainBiases )

  def test02_SingleLayerNoBiasControlled(self):

    # Trains a simple network with one single step, verifies
    # the training works as expected by calculating the same
    # as the trainer should do using python.
    machine = bob.machine.MLP((4, 1))
    machine.activation = bob.machine.Activation.LINEAR
    machine.biases = 0
    w0 = numpy.array([[.1],[.2],[-.1],[-.05]])
    machine.weights = [w0]
    trainer = bob.trainer.MLPRPropTrainer(machine, 1)
    trainer.trainBiases = False
    d0 = numpy.array([[1., 2., 0., 2.]])
    t0 = numpy.array([[1.]])

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = bob.machine.MLP(machine) #a copy
    pytrainer.train(pymachine, d0, t0)

    # trains with our C++ implementation
    trainer.train_(machine, d0, t0)
    self.assertTrue( numpy.array_equal(pymachine.weights[0], machine.weights[0]) )

    # a second passage
    d0 = numpy.array([[4., 0., -3., 1.]])
    t0 = numpy.array([[2.]])
    pytrainer.train(pymachine, d0, t0)
    trainer.train_(machine, d0, t0)
    self.assertTrue( numpy.array_equal(pymachine.weights[0], machine.weights[0]) )

    # a third passage
    d0 = numpy.array([[-0.5, -9.0, 2.0, 1.1]])
    t0 = numpy.array([[3.]])
    pytrainer.train(pymachine, d0, t0)
    trainer.train_(machine, d0, t0)
    self.assertTrue( numpy.array_equal(pymachine.weights[0], machine.weights[0]) )

  def test03_FisherNoBias(self):
    
    # Trains single layer MLP to discriminate the iris plants from
    # Fisher's paper. Checks we get a performance close to the one on
    # that paper.

    N = 60

    machine = bob.machine.MLP((4, 1))
    machine.activation = bob.machine.Activation.LINEAR
    machine.randomize()
    machine.biases = 0
    trainer = bob.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = False

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        numpy.array([-2.0]), #setosa
        numpy.array([1.5]), #versicolor
        numpy.array([0.5]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = bob.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = bob.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = bob.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(100):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] MSE:", bob.measure.mse(pymachine(input), target).sqrt()
      #print "[C++] MSE:", bob.measure.mse(machine(input), target).sqrt()
      self.assertTrue( numpy.array_equal(pymachine.weights[0], machine.weights[0]) )

  def test04_Fisher(self):
    
    # Trains single layer MLP to discriminate the iris plants from
    # Fisher's paper. Checks we get a performance close to the one on
    # that paper.

    N = 60

    machine = bob.machine.MLP((4, 1))
    machine.activation = bob.machine.Activation.LINEAR
    machine.randomize()
    trainer = bob.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = True

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        numpy.array([-2.0]), #setosa
        numpy.array([1.5]), #versicolor
        numpy.array([0.5]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = bob.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = bob.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = bob.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(100):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] MSE:", bob.measure.mse(pymachine(input), target).sqrt()
      #print "[C++] MSE:", bob.measure.mse(machine(input), target).sqrt()
      self.assertTrue( numpy.array_equal(pymachine.weights[0], machine.weights[0]) )
      self.assertTrue( numpy.array_equal(pymachine.biases[0], machine.biases[0]) )

  def test05_FisherWithOneHiddenLayer(self):

    # Trains a multilayer biased MLP to perform discrimination on the Fisher
    # data set.

    N = 50

    machine = bob.machine.MLP((4, 4, 3))
    machine.activation = bob.machine.Activation.TANH
    machine.randomize()
    trainer = bob.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = True

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        numpy.array([+1., -1., -1.]), #setosa
        numpy.array([-1., +1., -1.]), #versicolor
        numpy.array([-1., -1., +1.]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = bob.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = bob.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = bob.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(50):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] |RMSE|:", numpy.linalg.norm(bob.measure.rmse(pymachine(input), target))
      #print "[C++] |RMSE|:", numpy.linalg.norm(bob.measure.rmse(machine(input), target))
      for i, w in enumerate(pymachine.weights):
        self.assertTrue( numpy.array_equal(w, machine.weights[i]) )
      for i, b in enumerate(pymachine.biases):
        self.assertTrue( numpy.array_equal(b, machine.biases[i]) )

  def test06_FisherMultiLayer(self):

    # Trains a multilayer biased MLP to perform discrimination on the Fisher
    # data set.

    N = 50

    machine = bob.machine.MLP((4, 3, 3, 1))
    machine.activation = bob.machine.Activation.TANH
    machine.randomize()
    trainer = bob.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = True

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        numpy.array([-1.0]), #setosa
        numpy.array([0.5]), #versicolor
        numpy.array([+1.0]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = bob.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = bob.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = bob.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(50):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] MSE:", bob.measure.mse(pymachine(input), target).sqrt()
      #print "[C++] MSE:", bob.measure.mse(machine(input), target).sqrt()
      for i, w in enumerate(pymachine.weights):
        self.assertTrue( numpy.array_equal(w, machine.weights[i]) )
      for i, b in enumerate(pymachine.biases):
        self.assertTrue( numpy.array_equal(b, machine.biases[i]) )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(RPropTest)
