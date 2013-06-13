#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 13 Jun 2013 16:58:21 CEST
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

"""Tests on the MLP infrastructure.
"""

import numpy
import nose.tools
from .. import MLP, HyperbolicTangentActivation, LogisticActivation
from . import mlp as pymlp
from ... import io
from ...test import utils as test_utils

def test_2in_1out():

  m = MLP((2,1))
  assert m.shape == (2,1)
  assert m.input_divide.shape == (2,)
  assert m.input_subtract.shape == (2,)
  assert len(m.weights) == 1
  assert m.weights[0].shape == (2,1)
  assert numpy.allclose(m.weights[0], 0., rtol=1e-10, atol=1e-15)
  assert len(m.biases) == 1
  assert m.biases[0].shape == (1,)
  assert m.biases[0] == 0.
  assert m.hidden_activation == HyperbolicTangentActivation()
  assert m.output_activation == HyperbolicTangentActivation()

  # calculate and match
  weights = [numpy.random.rand(2,1)]
  biases = [numpy.random.rand(1)]

  m.weights = weights
  m.biases = biases

  pymac = pymlp.Machine(biases, weights, m.hidden_activation, m.output_activation)

  X = numpy.random.rand(10,2)
  assert numpy.allclose(m(X), pymac.forward(X), rtol=1e-10, atol=1e-15)

def test_2in_3_1out():
    
  m = MLP((2,3,1))
  assert m.shape == (2,3,1)
  assert m.input_divide.shape == (2,)
  assert m.input_subtract.shape == (2,)
  assert len(m.weights) == 2
  assert m.weights[0].shape == (2,3)
  assert numpy.allclose(m.weights[0], 0., rtol=1e-10, atol=1e-15)
  assert m.weights[1].shape == (3,1)
  assert numpy.allclose(m.weights[1], 0., rtol=1e-10, atol=1e-15)
  assert len(m.biases) == 2
  assert m.biases[0].shape == (3,)
  assert numpy.allclose(m.biases[0], 0., rtol=1e-10, atol=1e-15)
  assert m.biases[1].shape == (1,)
  assert numpy.allclose(m.biases[1], 0., rtol=1e-10, atol=1e-15)
  assert m.hidden_activation == HyperbolicTangentActivation()
  assert m.output_activation == HyperbolicTangentActivation()

  # calculate and match
  weights = [numpy.random.rand(2,3), numpy.random.rand(3,1)]
  biases = [numpy.random.rand(3), numpy.random.rand(1)]

  m.weights = weights
  m.biases = biases

  pymac = pymlp.Machine(biases, weights, m.hidden_activation, m.output_activation)

  X = numpy.random.rand(10,2)
  assert numpy.allclose(m(X), pymac.forward(X), rtol=1e-10, atol=1e-15)

def test_2in_3_5_1out():
    
  m = MLP((2,3,5,1))
  assert m.shape == (2,3,5,1)
  assert m.input_divide.shape == (2,)
  assert m.input_subtract.shape == (2,)
  assert len(m.weights) == 3
  assert m.weights[0].shape == (2,3)
  assert numpy.allclose(m.weights[0], 0., rtol=1e-10, atol=1e-15)
  assert m.weights[1].shape == (3,5)
  assert numpy.allclose(m.weights[1], 0., rtol=1e-10, atol=1e-15)
  assert m.weights[2].shape == (5,1)
  assert numpy.allclose(m.weights[2], 0., rtol=1e-10, atol=1e-15)
  assert len(m.biases) == 3
  assert m.biases[0].shape == (3,)
  assert numpy.allclose(m.biases[0], 0., rtol=1e-10, atol=1e-15)
  assert m.biases[1].shape == (5,)
  assert numpy.allclose(m.biases[1], 0., rtol=1e-10, atol=1e-15)
  assert m.biases[2].shape == (1,)
  assert numpy.allclose(m.biases[2], 0., rtol=1e-10, atol=1e-15)
  assert m.hidden_activation == HyperbolicTangentActivation()
  assert m.output_activation == HyperbolicTangentActivation()

  # calculate and match
  weights = [
      numpy.random.rand(2,3), 
      numpy.random.rand(3,5),
      numpy.random.rand(5,1)
      ]
  biases = [
      numpy.random.rand(3), 
      numpy.random.rand(5), 
      numpy.random.rand(1),
      ]

  m.weights = weights
  m.biases = biases

  pymac = pymlp.Machine(biases, weights, m.hidden_activation, m.output_activation)

  X = numpy.random.rand(10,2)
  assert numpy.allclose(m(X), pymac.forward(X), rtol=1e-10, atol=1e-15)

def test_100in_100_10_4out():
    
  m = MLP((100,100,10,4))

  # calculate and match
  weights = [
      numpy.random.rand(100,100), 
      numpy.random.rand(100,10),
      numpy.random.rand(10,4)
      ]
  biases = [
      numpy.random.rand(100), 
      numpy.random.rand(10), 
      numpy.random.rand(4),
      ]

  m.weights = weights
  m.biases = biases

  pymac = pymlp.Machine(biases, weights, m.hidden_activation, m.output_activation)

  X = numpy.random.rand(20,100)
  assert numpy.allclose(m(X), pymac.forward(X), rtol=1e-10, atol=1e-15)
def test_resize():
    
  m = MLP((2,3,5,1))
  m.shape = (2,1)
  m.hidden_activation = LogisticActivation()
  m.output_activation = LogisticActivation()

  assert m.shape == (2,1)
  assert m.input_divide.shape == (2,)
  assert m.input_subtract.shape == (2,)
  assert len(m.weights) == 1
  assert m.weights[0].shape == (2,1)
  assert numpy.allclose(m.weights[0], 0., rtol=1e-10, atol=1e-15)
  assert len(m.biases) == 1
  assert m.biases[0].shape == (1,)
  assert m.biases[0] == 0.
  assert m.hidden_activation == LogisticActivation()
  assert m.output_activation == LogisticActivation()

  # calculate and match
  weights = [numpy.random.rand(2,1)]
  biases = [numpy.random.rand(1)]

  m.weights = weights
  m.biases = biases

  pymac = pymlp.Machine(biases, weights, m.hidden_activation, m.output_activation)

  X = numpy.random.rand(10,2)
  assert numpy.allclose(m(X), pymac.forward(X), rtol=1e-10, atol=1e-15)

def test_checks():

  # tests if MLPs check wrong settings
  m = MLP((2,1))

  # the MLP shape cannot have a single entry
  nose.tools.assert_raises(RuntimeError, setattr, m, 'shape', (5,))

  # you cannot set the weights vector with the wrong size
  nose.tools.assert_raises(RuntimeError,
      setattr, m, 'weights', [numpy.zeros((3,1), 'float64')])

  # the same for the bias
  nose.tools.assert_raises(RuntimeError,
      setattr, m, 'biases', [numpy.zeros((5,), 'float64')])
  
  # it works though if the sizes are correct
  new_weights = [numpy.zeros((2,1), 'float64')]
  new_weights[0].fill(3.14)
  m.weights = new_weights

  assert len(m.weights) == 1

  assert (m.weights[0] == new_weights[0]).all()

  new_biases = [numpy.zeros((1,), 'float64')]
  new_biases[0].fill(5.71)
  m.biases = new_biases

  assert len(m.biases) == 1

  assert (m.biases[0] == new_biases[0]).all()

def test_persistence():

  # make shure we can save an load an MLP machine
  weights = []
  weights.append(numpy.array([[.2, -.1, .2], [.2, .3, .9]]))
  weights.append(numpy.array([[.1, .5], [-.1, .2], [-.1, 1.1]])) 
  biases = []
  biases.append(numpy.array([-.1, .3, .1]))
  biases.append(numpy.array([.2, -.1]))
  
  m = MLP((2,3,2))
  m.weights = weights
  m.biases = biases

  # creates a file that will be used in the next test!
  machine_file = test_utils.temporary_filename()
  m.save(io.HDF5File(machine_file, 'w'))
  m2 = MLP(io.HDF5File(machine_file))
  
  assert m.is_similar_to(m2)
  assert m == m2
  assert m.shape == m2.shape
  assert (m.input_subtract == m2.input_subtract).all()
  assert (m.input_divide == m2.input_divide).all()
  
  for i in range(len(m.weights)):
    assert (m.weights[i] == m2.weights[i]).all()
    assert (m.biases[i] == m2.biases[i]).all()

def test_randomization():

  m = MLP((2,3,2))
  m.randomize()

  for k in m.weights:
    assert (abs(k) <= 0.1).all()
    assert (k != 0).any()

  for k in m.biases:
    assert (abs(k) <= 0.1).all()
    assert (k != 0).any()

def test_randomization_margins():

  # we can also reset the margins for randomization
  for k in range(10):

    m = MLP((2,3,2))
    m.randomize(-0.001, 0.001)

    for k in m.weights:
      assert (abs(k) <= 0.001).all()
      assert (k != 0).any()

    for k in m.biases:
      assert (abs(k) <= 0.001).all()
      assert (k != 0).any()

def test_randomness():

  m1 = MLP((2,3,2))
  m1.randomize()

  for k in range(10):
    m2 = MLP((2,3,2))
    m2.randomize()

    for w1, w2 in zip(m1.weights, m2.weights):
      assert (w1 == w2).all() == False

    for b1, b2 in zip(m1.biases, m2.biases):
      assert (b1 == b2).all() == False
