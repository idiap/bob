#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Jul 19 09:47:23 2011 +0200
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

"""Tests for BackProp MLP training.
"""

import os, sys
import math
import unittest
import numpy

from .. import MLPBaseTrainer, CrossEntropyLoss, SquareError
from ...machine import HyperbolicTangentActivation, LogisticActivation, IdentityActivation

class PythonBackProp(MLPBaseTrainer):
  """A simple version of the vanilla BackProp algorithm written in Python
  """

  def __init__(self, batch_size, cost, machine, train_biases, 
      learning_rate=0.1, momentum=0.0):
    
    super(self, PythonBackProp).__init__(batch_size, cost, machine, train_biases)
    self.prev_deriv = [numpy.zeros(k.shape, dtype=float) for k in machine.weights]
    self.prev_deriv_bias = [numpy.zeros(k.shape, dtype=float) for k in machine.biases]

  def train(self, machine, input, target):

    # Run dataset through
    prev_cost = self.cost(machine, input, target)
    self.backward_step(machine, input, target)

    # Updates weights and biases
    new_weights = machine.weights
    for k,W in enumerate(new_weights):
      new_weights[k] = W + (((1-self.momentum)*self.deriv[k]) + (self.momentum*self.prev_deriv[k])) for k in range(len(W))]
    self.prev_deriv = self.deriv
    machine.weights = new_weights

    if self.train_biases:
      new_biases = machine.biases
      for k,B in enumerate(new_biases):
        new_biases[k] = [B[k] + (((1-self.momentum)*self.deriv_bias[k]) + (self.momentum*self.prev_deriv_bias[k])) for k in range(len(W))]
      self.prev_deriv_bias = self.deriv_bias
      machine.biases = new_biases

def test_2in_1out_nobias():

  machine = bob.machine.MLP((2, 1))
  machine.randomize()
  machine.biases = 0

  cxx_trainer = bob.trainer.MLPBackPropTrainer(1, bob.trainer.SquareError(machine.output_activation))
  trainer.train_biases = False
  trainer.initialize(machine)
  d0 = numpy.array([[.3, .7]])
  t0 = numpy.array([[.0]])

  # trains in python first
  pytrainer = PythonBackProp(train_biases=trainer.train_biases)
  pymachine = bob.machine.MLP(machine) #a copy
  pytrainer.train(pymachine, d0, t0)

  # trains with our C++ implementation
  trainer.train_(machine, d0, t0)
  self.assertTrue(numpy.array_equal(pymachine.weights[0], machine.weights[0]))

def test03_FisherWithOneHiddenLayer(self):

  # Trains a multilayer biased MLP to perform discrimination on the Fisher
  # data set.

  N = 50

  machine = bob.machine.MLP((4, 4, 3))
  machine.hidden_activation = bob.machine.HyperbolicTangentActivation()
  machine.output_activation = bob.machine.HyperbolicTangentActivation()
  machine.randomize()
  trainer = bob.trainer.MLPBackPropTrainer(N, bob.trainer.SquareError(machine.output_activation), machine)
  trainer.train_biases = True

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
  pytrainer = PythonBackProp(train_biases=trainer.train_biases)
  pymachine = bob.machine.MLP(machine) #a copy

  # We now iterate for several steps, look for the convergence
  for k in range(50):
    input, target = S(N)
    pytrainer.train(pymachine, input, target)
    trainer.train_(machine, input, target)
    #print "[Python] |RMSE|@%d:" % k, numpy.linalg.norm(bob.measure.rmse(pymachine(input), target))
    #print "[C++] |RMSE|@%d:" % k, numpy.linalg.norm(bob.measure.rmse(machine(input), target))
    # Note we will face precision problems when comparing to the Pythonic
    # implementation. So, let's not be too demanding here. If all values are
    # approximately equal to 1e-10, we consider this is OK.
    for i, w in enumerate(pymachine.weights):
      self.assertTrue( (abs(w-machine.weights[i]) < 1e-10).all() )
    for i, b in enumerate(pymachine.biases):
      self.assertTrue( (abs(b-machine.biases[i]) < 1e-10).all() )

def test04_FisherMultiLayer(self):

  # Trains a multilayer biased MLP to perform discrimination on the Fisher
  # data set.

  N = 50

  machine = bob.machine.MLP((4, 3, 3, 1))
  machine.hidden_activation = bob.machine.HyperbolicTangentActivation()
  machine.output_activation = bob.machine.HyperbolicTangentActivation()
  machine.randomize()
  trainer = bob.trainer.MLPBackPropTrainer(N, bob.trainer.SquareError(machine.output_activation))
  trainer.train_biases = True
  trainer.initialize(machine)

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
  pytrainer = PythonBackProp(train_biases=trainer.train_biases)
  pymachine = bob.machine.MLP(machine) #a copy

  # We now iterate for several steps, look for the convergence
  for k in range(50):
    input, target = S(N)
    pytrainer.train(pymachine, input, target)
    trainer.train_(machine, input, target)
    #print "[Python] MSE:", bob.measure.mse(pymachine(input), target).sqrt()
    #print "[C++] MSE:", bob.measure.mse(machine(input), target).sqrt()
    # Note we will face precision problems when comparing to the Pythonic
    # implementation. So, let's not be too demanding here. If all values are
    # approximately equal to 1e-10, we consider this is OK.
    for i, w in enumerate(pymachine.weights):
      self.assertTrue( (abs(w-machine.weights[i]) < 1e-10).all() )
    for i, b in enumerate(pymachine.biases):
      self.assertTrue( (abs(b-machine.biases[i]) < 1e-10).all() )

def test05_FisherMultiLayerWithMomentum(self):

  # Trains a multilayer biased MLP to perform discrimination on the Fisher
  # data set.

  N = 50

  machine = bob.machine.MLP((4, 3, 3, 1))
  machine.hidden_activation = bob.machine.HyperbolicTangentActivation()
  machine.output_activation = bob.machine.HyperbolicTangentActivation()
  machine.randomize()
  trainer = bob.trainer.MLPBackPropTrainer(N, bob.trainer.SquareError(machine.output_activation), machine)
  trainer.train_biases = True
  trainer.momentum = 0.99

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
  pytrainer = PythonBackProp(train_biases=trainer.train_biases)
  pymachine = bob.machine.MLP(machine) #a copy
  pytrainer.momentum = 0.99

  # We now iterate for several steps, look for the convergence
  for k in range(50):
    input, target = S(N)
    pytrainer.train(pymachine, input, target)
    trainer.train_(machine, input, target)
    #print "[Python] MSE:", bob.measure.mse(pymachine(input), target).sqrt()
    #print "[C++] MSE:", bob.measure.mse(machine(input), target).sqrt()
    # Note we will face precision problems when comparing to the Pythonic
    # implementation. So, let's not be too demanding here. If all values are
    # approximately equal to 1e-10, we consider this is OK.
    for i, w in enumerate(pymachine.weights):
      self.assertTrue( (abs(w-machine.weights[i]) < 1e-10).all() )
    for i, b in enumerate(pymachine.biases):
      self.assertTrue( (abs(b-machine.biases[i]) < 1e-10).all() )
