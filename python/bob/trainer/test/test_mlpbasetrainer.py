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
import numpy

from .mlp import Machine as PythonMachine
from . import gradient
from .. import MLPBaseTrainer, CrossEntropyLoss, SquareError
from ...machine import MLP, LogisticActivation, HyperbolicTangentActivation

def python_check_rolling(machine, bias_training):


  pymac = PythonMachine(machine.biases if bias_training else None,
      machine.weights, machine.hidden_activation, machine.output_activation)

  unrolled_weights = pymac.unroll()
  rolled_weights = pymac.roll(unrolled_weights)

  for k,w in enumerate(pymac.weights):
    assert numpy.alltrue(rolled_weights[k] == w)

def python_check_gradient(machine, cost, bias_training, batch_size):

  pymac = PythonMachine(machine.biases if bias_training else None,
      machine.weights, machine.hidden_activation, machine.output_activation)

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

def cxx_vs_python_check_gradient(machine, cost, bias_training, batch_size):

  pymac = PythonMachine(machine.biases if bias_training else None,
      machine.weights, machine.hidden_activation, machine.output_activation)

  X = numpy.random.rand(batch_size, machine.weights[0].shape[0])
  T = numpy.zeros((batch_size, machine.weights[-1].shape[1]))

  # use the pythonic (tested) infrastructure to calculate the gradient
  b = cost.error(pymac.forward(X), T)
  pythonic = pymac.backward(b)

  # use the C++ infrastructure to calculate the gradient
  trainer = MLPBaseTrainer(batch_size, cost, machine)
  trainer.forward_step(machine, X)
  trainer.backward_step(machine, X, T)

  expected_precision = 1e-10

  for k,(dw,db) in enumerate(zip(trainer.deriv, trainer.deriv_bias)):
    if bias_training: dw = numpy.vstack((db, dw))
    absdiff = abs((pythonic[k]-dw)/dw)
    pos = absdiff.argmax()
    pos = (pos/dw.shape[1], pos%dw.shape[1])
    assert numpy.alltrue(absdiff < expected_precision), "Maximum relative difference in layer %d is greater than %g - happens for element at position %s (calculated = %g; python = %g)" % (k, expected_precision, pos, dw[pos], pythonic[k][pos])

def test_2in_1out_nobias():
  
  machine = MLP((2, 1))
  machine.randomize()
  machine.hidden_activation = LogisticActivation()
  machine.output_activation = LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = CrossEntropyLoss(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)
  cxx_vs_python_check_gradient(machine, cost, False, BATCH_SIZE)

def test_1in_2out_nobias():

  machine = MLP((1, 2))
  machine.randomize()
  machine.hidden_activation = LogisticActivation()
  machine.output_activation = LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = CrossEntropyLoss(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)
  cxx_vs_python_check_gradient(machine, cost, False, BATCH_SIZE)
  
def test_2in_3_1out_nobias():

  machine = MLP((2, 3, 1))
  machine.randomize()
  machine.hidden_activation = LogisticActivation()
  machine.output_activation = LogisticActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = CrossEntropyLoss(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)
  cxx_vs_python_check_gradient(machine, cost, False, BATCH_SIZE)
  
def test_100in_10_10_5out_nobias():

  machine = MLP((100, 10, 10, 5))
  machine.randomize()
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()
  machine.biases = 0

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  python_check_rolling(machine, False)
  python_check_gradient(machine, cost, False, BATCH_SIZE)
  cxx_vs_python_check_gradient(machine, cost, False, BATCH_SIZE)

def test_2in_3_1out():

  machine = MLP((2, 3, 1))
  machine.hidden_activation = LogisticActivation()
  machine.output_activation = LogisticActivation()
  machine.randomize()

  BATCH_SIZE = 10
  cost = CrossEntropyLoss(machine.output_activation)

  python_check_rolling(machine, True)
  python_check_gradient(machine, cost, True, BATCH_SIZE)
  cxx_vs_python_check_gradient(machine, cost, True, BATCH_SIZE)

def test_20in_10_5_3out():

  machine = MLP((20, 10, 5, 3))
  machine.hidden_activation = HyperbolicTangentActivation()
  machine.output_activation = HyperbolicTangentActivation()
  machine.randomize()

  BATCH_SIZE = 10
  cost = SquareError(machine.output_activation)

  python_check_rolling(machine, True)
  python_check_gradient(machine, cost, True, BATCH_SIZE)
  cxx_vs_python_check_gradient(machine, cost, True, BATCH_SIZE)
