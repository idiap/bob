#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Jul 14 18:53:07 2011 +0200
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

"""Tests for RProp MLP training.
"""

import numpy

from .. import MLPBaseTrainer, MLPRPropTrainer, CrossEntropyLoss, SquareError
from ...machine import HyperbolicTangentActivation, LogisticActivation, IdentityActivation, MLP

def sign(x):
  """A handy sign function"""
  if (x == 0): return 0
  if (x < 0) : return -1
  return +1

class PythonRProp(MLPBaseTrainer):
  """A simple version of the R-Prop algorithm written in Python
  """

  def __init__(self, batch_size, cost, machine, train_biases):
    
    super(PythonRProp, self).__init__(batch_size, cost, machine, train_biases)

    # some constants for RProp
    self.DELTA0 = 0.1
    self.DELTA_MIN = 1e-6
    self.DELTA_MAX = 50
    self.ETA_MINUS = 0.5
    self.ETA_PLUS = 1.2

    # initial derivatives
    self.previous_derivatives = [numpy.zeros(k.shape, dtype=float) for k in machine.weights]
    self.previous_bias_derivatives = [numpy.zeros(k.shape, dtype=float) for k in machine.biases]

    # initial deltas
    self.deltas = [self.DELTA0*numpy.ones(k.shape, dtype=float) for k in machine.weights]
    self.bias_deltas = [self.DELTA0*numpy.ones(k.shape, dtype=float) for k in machine.biases]

  def train(self, machine, input, target):

    # Run dataset through
    prev_cost = self.cost(machine, input, target)
    self.backward_step(machine, input, target)

    # Updates weights and biases
    weight_updates = [i * j for (i,j) in zip(self.previous_derivatives, self.derivatives)]

    # Iterate over each weight and bias and see what to do:
    new_weights = machine.weights
    for k, up in enumerate(weight_updates):
      for i in range(up.shape[0]):
        for j in range(up.shape[1]):
          if up[i,j] > 0:
            self.deltas[k][i,j] = min(self.deltas[k][i,j]*ETA_PLUS, DELTA_MAX)
            new_weights[k][i,j] -= sign(self.derivatives[k][i,j]) * self.deltas[k][i,j]
            self.previous_derivatives[k][i,j] = self.derivatives[k][i,j]
          elif up[i,j] < 0:
            self.deltas[k][i,j] = max(self.deltas[k][i,j]*ETA_MINUS, DELTA_MIN)
            new_weights[k][i,j] -= self.deltas[k][i,j]
            self.previous_derivatives[k][i,j] = 0
          else:
            new_weights[k][i,j] -= sign(self.derivatives[k][i,j]) * self.deltas[k][i,j]
            self.previous_derivatives[k][i,j] = self.derivatives[k][i,j]
    machine.weights = new_weights

    if self.train_biases:
      bias_updates = [i * j for (i,j) in zip(self.previous_bias_derivatives, self.bias_derivatives)]
      new_biases = machine.biases
      for k, up in enumerate(bias_updates):
        for i in range(up.shape[0]):
          if up[i] > 0:
            self.bias_deltas[k][i] = min(self.bias_deltas[k][i]*ETA_PLUS, DELTA_MAX)
            new_biases[k][i] -= sign(self.bias_derivatives[k][i]) * self.bias_deltas[k][i]
            self.previous_bias_derivatives[k][i] = self.bias_derivatives[k][i]
          elif up[i] < 0:
            self.bias_deltas[k][i] = max(self.bias_deltas[k][i]*ETA_MINUS, DELTA_MIN)
            new_biases[k][i] -= self.bias_deltas[k][i]
            self.previous_bias_derivatives[k][i] = 0
          else:
            new_biases[k][i] -= sign(self.bias_derivatives[k][i]) * self.bias_deltas[k][i]
            self.previous_bias_derivatives[k][i] = self.bias_derivatives[k][i]
      machine.biases = new_biases

    else:
      machine.biases = 0

    return prev_cost, self.cost(machine, input, target)

def check_training(machine, cost, bias_training, batch_size):

  python_machine = MLP(machine)
  
  X = numpy.random.rand(batch_size, machine.weights[0].shape[0])
  T = numpy.zeros((batch_size, machine.weights[-1].shape[1]))

  python_trainer = PythonRProp(batch_size, cost, machine, bias_training)
  cxx_trainer = MLPRPropTrainer(batch_size, cost, machine, bias_training)

  # checks previous state matches
  for k,D in enumerate(cxx_trainer.deltas):
    assert numpy.allclose(D, python_trainer.deltas[k])
  for k,D in enumerate(cxx_trainer.bias_deltas):
    assert numpy.allclose(D, python_trainer.bias_deltas[k])
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
  #assert cost <= prev_cost #not true for R-Prop
  cxx_trainer.train(machine, X, T)

  # checks each component of machine and trainer, make sure they match
  for k,D in enumerate(cxx_trainer.deltas):
    assert numpy.allclose(D, python_trainer.deltas[k])
  for k,D in enumerate(cxx_trainer.bias_deltas):
    assert numpy.allclose(D, python_trainer.bias_deltas[k])
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
    check_training(machine, cost, False, BATCH_SIZE)

def test_1in_2out_nobias():

  machine = MLP((1, 2))
  machine.randomize()
  machine.hidden_activation = LogisticActivation()
  machine.output_activation = LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = CrossEntropyLoss(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, False, BATCH_SIZE)

def test_2in_3_1out_nobias():

  machine = MLP((2, 3, 1))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, False, BATCH_SIZE)

def test_100in_10_10_5out_nobias():

  machine = MLP((100, 10, 10, 5))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, False, BATCH_SIZE)

def test_2in_3_1out():

  machine = MLP((2, 3, 1))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, True, BATCH_SIZE)

def test_20in_10_5_3out():

  machine = MLP((20, 10, 5, 3))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, True, BATCH_SIZE)

def test_20in_10_5_3out_with_momentum():

  machine = MLP((20, 10, 5, 3))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  for k in range(10):
    check_training(machine, cost, True, BATCH_SIZE)
