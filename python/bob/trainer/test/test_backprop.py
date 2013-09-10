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

import numpy

from .. import MLPBaseTrainer, MLPBackPropTrainer, CrossEntropyLoss, SquareError
from ...machine import HyperbolicTangentActivation, LogisticActivation, IdentityActivation, MLP

class PythonBackProp(MLPBaseTrainer):
  """A simple version of the vanilla BackProp algorithm written in Python
  """

  def __init__(self, batch_size, cost, machine, train_biases, 
      learning_rate=0.1, momentum=0.0):
    
    super(PythonBackProp, self).__init__(batch_size, cost, machine, train_biases)
    self.previous_derivatives = [numpy.zeros(k.shape, dtype=float) for k in machine.weights]
    self.previous_bias_derivatives = [numpy.zeros(k.shape, dtype=float) for k in machine.biases]

    self.learning_rate = learning_rate
    self.momentum = momentum

  def train(self, machine, input, target):

    # Run dataset through
    prev_cost = self.cost(machine, input, target)
    self.backward_step(machine, input, target)

    # Updates weights and biases
    new_weights = list(machine.weights)
    for k,W in enumerate(new_weights):
      new_weights[k] = W - (((1-self.momentum)*self.learning_rate*self.derivatives[k]) + (self.momentum*self.previous_derivatives[k]))
    self.previous_derivatives = [self.learning_rate*k for k in self.derivatives]
    machine.weights = new_weights

    if self.train_biases:
      new_biases = list(machine.biases)
      for k,B in enumerate(new_biases):
        new_biases[k] = B - (((1-self.momentum)*self.learning_rate*self.bias_derivatives[k]) + (self.momentum*self.previous_bias_derivatives[k]))
      self.previous_bias_derivatives = [self.learning_rate*k for k in self.bias_derivatives]
      machine.biases = new_biases

    return prev_cost, self.cost(machine, input, target)

def check_training(machine, cost, bias_training, batch_size, learning_rate,
    momentum):

  python_machine = MLP(machine)
  
  X = numpy.random.rand(batch_size, machine.weights[0].shape[0])
  T = numpy.zeros((batch_size, machine.weights[-1].shape[1]))

  python_trainer = PythonBackProp(batch_size, cost, machine, bias_training,
      learning_rate, momentum)
  cxx_trainer = MLPBackPropTrainer(batch_size, cost, machine, bias_training)
  cxx_trainer.learning_rate = learning_rate
  cxx_trainer.momentum = momentum

  # checks previous state matches
  for k,D in enumerate(cxx_trainer.previous_derivatives):
    assert numpy.allclose(D, python_trainer.previous_derivatives[k])
  for k,D in enumerate(cxx_trainer.previous_bias_derivatives):
    assert numpy.allclose(D, python_trainer.previous_bias_derivatives[k])
  for k,W in enumerate(machine.weights):
    assert numpy.allclose(W, python_machine.weights[k])
  for k,B in enumerate(machine.biases):
    assert numpy.allclose(B, python_machine.biases[k])
  assert numpy.alltrue(machine.input_subtract == python_machine.input_subtract)
  assert numpy.alltrue(machine.input_divide == python_machine.input_divide)

  prev_cost, cost = python_trainer.train(python_machine, X, T)
  assert cost <= prev_cost #this should always be true for a fixed dataset
  cxx_trainer.train(machine, X, T)

  # checks each component of machine and trainer, make sure they match
  for k,D in enumerate(cxx_trainer.derivatives):
    assert numpy.allclose(D, python_trainer.derivatives[k])
  for k,D in enumerate(cxx_trainer.bias_derivatives):
    assert numpy.allclose(D, python_trainer.bias_derivatives[k])
  for k,W in enumerate(machine.weights):
    assert numpy.allclose(W, python_machine.weights[k])
  for k,B in enumerate(machine.biases):
    assert numpy.allclose(B, python_machine.biases[k])
  assert numpy.alltrue(machine.input_subtract == python_machine.input_subtract)
  assert numpy.alltrue(machine.input_divide == python_machine.input_divide)

def test_2in_1out_nobias():
  
  machine = MLP((2, 1))
  machine.randomize()
  machine.hidden_activation = LogisticActivation()
  machine.output_activation = LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = CrossEntropyLoss(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, False, BATCH_SIZE, 0.1, 0.0)

def test_1in_2out_nobias():

  machine = MLP((1, 2))
  machine.randomize()
  machine.hidden_activation = LogisticActivation()
  machine.output_activation = LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = CrossEntropyLoss(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, False, BATCH_SIZE, 0.1, 0.0)

def test_2in_3_1out_nobias():

  machine = MLP((2, 3, 1))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, False, BATCH_SIZE, 0.1, 0.0)

def test_100in_10_10_5out_nobias():

  machine = MLP((100, 10, 10, 5))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, False, BATCH_SIZE, 0.1, 0.0)

def test_2in_3_1out():

  machine = MLP((2, 3, 1))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, True, BATCH_SIZE, 0.1, 0.0)

def test_20in_10_5_3out():

  machine = MLP((20, 10, 5, 3))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, True, BATCH_SIZE, 0.1, 0.0)

def test_20in_10_5_3out_with_momentum():

  machine = MLP((20, 10, 5, 3))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, True, BATCH_SIZE, 0.1, 0.1)
