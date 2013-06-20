#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 20 Jun 14:56:56 2013 CEST 
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

"""Tests for the base MLP trainer.
"""

import os, sys
import math
import bob
import numpy

from .mlp import Machine
from . import gradient

def python_check_rolling(machine, bias_training):


  pymac = Machine(machine.bias if bias_training else None, machine.weights, 
      machine.hidden_activation, machine.output_activation)

  unrolled_weights = pymac.unroll()
  rolled_weights = pymac.roll(unrolled_weights)

  for k,w in enumerate(pymac.weights):
    assert numpy.alltrue(rolled_weights[k] == w)

def python_check_gradient(machine, cost, bias_training, batch_size):

  pymac = Machine(machine.bias if bias_training else None, machine.weights, 
      machine.hidden_activation, machine.output_activation)

  X = numpy.random.rand(batch_size, machine.weights[0].shape[0])
  T = numpy.zeros((batch_size, machine.weights[-1].shape[1]))

  # make sure our pythonic implementation has the correct gradient estimation
  b = cost.error(pymac.forward(X), T)
  derived = pymac.backward(b)
  estimated = gradient.estimate_for_machine(pymac, X, cost, T)

  expected_precision = 1e-3

  for k,d in enumerate(derived):
    absdiff = abs((d-estimated[k])/d)
    pos = absdiff.argmax()
    pos = (pos/d.shape[1], pos%d.shape[1])
    assert numpy.alltrue(absdiff < expected_precision), "Maximum relative difference in layer %d is greater than %g - happens for element at position %s (calculated = %g; estimated = %g)" % (k, expected_precision, pos, d[pos], estimated[k][pos])

def test_python_2in_1out_nobias():
  
  machine = bob.machine.MLP((2, 1))
  machine.randomize()
  machine.hidden_activation = bob.machine.LogisticActivation()
  machine.output_activation = bob.machine.LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = bob.trainer.CrossEntropyLoss(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)

def test_python_1in_2out_nobias():

  machine = bob.machine.MLP((1, 2))
  machine.randomize()
  machine.hidden_activation = bob.machine.LogisticActivation()
  machine.output_activation = bob.machine.LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = bob.trainer.CrossEntropyLoss(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)
  
def test_python_2in_3_1out_nobias():

  machine = bob.machine.MLP((2, 3, 1))
  machine.randomize()
  machine.hidden_activation = bob.machine.LogisticActivation()
  machine.output_activation = bob.machine.LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = bob.trainer.CrossEntropyLoss(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)
  
def test_python_100in_100_10_5out_nobias():

  machine = bob.machine.MLP((100, 100, 10, 5))
  machine.randomize()
  machine.hidden_activation = bob.machine.HyperbolicTangentActivation()
  machine.output_activation = bob.machine.HyperbolicTangentActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = bob.trainer.SquareError(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)
