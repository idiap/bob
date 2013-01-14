#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Jul 8 09:40:22 2011 +0200
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

import os, sys
import unittest
import math
import numpy
import bob
import tempfile
import pkg_resources

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def tempname(suffix, prefix='bobtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

MACHINE = tempname(".hdf5")
COMPLICATED = F('mlp-big.hdf5')
COMPLICATED_OUTPUT = F('network.hdf5')
COMPLICATED_NOBIAS = F('mlp-big-nobias.hdf5')
COMPLICATED_NOBIAS_OUTPUT = F('network-without-bias.hdf5')

class MLPTest(unittest.TestCase):
  """Performs various MLP tests."""

  def test01_Initialization(self):

    # Two inputs and 1 output
    m = bob.machine.MLP((2,1))
    self.assertEqual(m.shape, (2,1))
    self.assertEqual(m.input_divide.shape[0], 2)
    self.assertEqual(m.input_subtract.shape[0], 2)
    self.assertEqual(len(m.weights), 1)
    self.assertEqual(m.weights[0].shape, (2,1))
    self.assertTrue((m.weights[0] == 0.0).all())
    self.assertEqual(len(m.biases), 1)
    self.assertEqual(m.biases[0].shape, (1,))
    self.assertTrue((m.biases[0] == 0.0).all())
    self.assertEqual(m.activation, bob.machine.Activation.TANH)

    # 1 hidden layer
    m = bob.machine.MLP((2,3,1))
    self.assertEqual(m.shape, (2,3,1))
    self.assertEqual(m.input_divide.shape[0], 2)
    self.assertEqual(m.input_subtract.shape[0], 2)
    self.assertEqual(len(m.weights), 2)
    self.assertEqual(m.weights[0].shape, (2,3))
    self.assertTrue((m.weights[0] == 0.0).all())
    self.assertEqual(m.weights[1].shape, (3,1))
    self.assertTrue((m.weights[1] == 0.0).all())
    self.assertEqual(len(m.biases), 2)
    self.assertEqual(m.biases[0].shape, (3,))
    self.assertTrue((m.biases[0] == 0.0).all())
    self.assertEqual(m.biases[1].shape, (1,))
    self.assertTrue((m.biases[1] == 0.0).all())
    self.assertEqual(m.activation, bob.machine.Activation.TANH)

    # 2+ hidden layers, different activation
    m = bob.machine.MLP((2,3,5,1))
    m.activation = bob.machine.Activation.LOG
    self.assertEqual(m.shape, (2,3,5,1))
    self.assertEqual(m.input_divide.shape[0], 2)
    self.assertEqual(m.input_subtract.shape[0], 2)
    self.assertEqual(len(m.weights), 3)
    self.assertEqual(m.weights[0].shape, (2,3))
    self.assertTrue((m.weights[0] == 0.0).all())
    self.assertEqual(m.weights[1].shape, (3,5))
    self.assertTrue((m.weights[1] == 0.0).all())
    self.assertEqual(m.weights[2].shape, (5,1))
    self.assertTrue((m.weights[2] == 0.0).all())
    self.assertEqual(len(m.biases), 3)
    self.assertEqual(m.biases[0].shape, (3,))
    self.assertTrue((m.biases[0] == 0.0).all())
    self.assertEqual(m.biases[1].shape, (5,))
    self.assertTrue((m.biases[1] == 0.0).all())
    self.assertEqual(m.biases[2].shape, (1,))
    self.assertTrue((m.biases[2] == 0.0).all())
    self.assertEqual(m.activation, bob.machine.Activation.LOG)

    # A resize should make the last machine look, structurally,
    # like the first again
    m.shape = (2,1)
    self.assertEqual(m.shape, (2,1))
    self.assertEqual(m.input_divide.shape[0], 2)
    self.assertEqual(m.input_subtract.shape[0], 2)
    self.assertEqual(len(m.weights), 1)
    self.assertEqual(m.weights[0].shape, (2,1))
    self.assertEqual(len(m.biases), 1)
    self.assertEqual(m.biases[0].shape, (1,))
    self.assertEqual(m.activation, bob.machine.Activation.LOG)

  def test02_Checks(self):

    # tests if MLPs check wrong settings
    m = bob.machine.MLP((2,1))

    # the MLP shape cannot have a single entry
    self.assertRaises(RuntimeError, setattr, m, 'shape', (5,))

    # you cannot set the weights vector with the wrong size
    self.assertRaises(RuntimeError,
        setattr, m, 'weights', [numpy.zeros((3,1), 'float64')])

    # the same for the bias
    self.assertRaises(RuntimeError,
        setattr, m, 'biases', [numpy.zeros((5,), 'float64')])
    
    # it works though if the sizes are correct
    new_weights = [numpy.zeros((2,1), 'float64')]
    new_weights[0].fill(3.14)
    m.weights = new_weights
    self.assertEqual(len(m.weights), 1)
    self.assertTrue( (m.weights[0] == new_weights[0]).all() )

    new_biases = [numpy.zeros((1,), 'float64')]
    new_biases[0].fill(5.71)
    m.biases = new_biases
    self.assertEqual(len(m.biases), 1)
    self.assertTrue( (m.biases[0] == new_biases[0]).all() )

  def test03_LoadingAndSaving(self):

    # make shure we can save an load an MLP machine
    weights = []
    weights.append(numpy.array([[.2, -.1, .2], [.2, .3, .9]]))
    weights.append(numpy.array([[.1, .5], [-.1, .2], [-.1, 1.1]])) 
    biases = []
    biases.append(numpy.array([-.1, .3, .1]))
    biases.append(numpy.array([.2, -.1]))
    
    m = bob.machine.MLP((2,3,2))
    m.weights = weights
    m.biases = biases

    # creates a file that will be used in the next test!
    m.save(bob.io.HDF5File(MACHINE, 'w'))
    m2 = bob.machine.MLP(bob.io.HDF5File(MACHINE))
    
    self.assertEqual(m.shape, m2.shape)
    self.assertTrue((m.input_subtract == m2.input_subtract).all())
    self.assertTrue((m.input_divide == m2.input_divide).all())
    for i in range(len(m.weights)):
      self.assertTrue((m.weights[i] == m2.weights[i]).all())
      self.assertTrue((m.biases[i] == m2.biases[i]).all())

  def test04_Correctness(self):

    # makes sure the outputs of the MLP are correct
    m = bob.machine.MLP(bob.io.HDF5File(MACHINE))
    i = numpy.array([.1, .7])
    y = m(i)
    y_exp = numpy.array([0.09596993, 0.6175601])
    self.assertTrue( (abs(y - y_exp) < 1e-6).all() )

    # compares a simple (logistic activation, 1 layer) MLP with a LinearMachine
    mlinear = bob.machine.LinearMachine(2,1)
    mlinear.activation = bob.machine.Activation.LOG
    mlinear.weights = numpy.array([[.3], [-.42]])
    mlinear.biases = numpy.array([-.7])

    mlp = bob.machine.MLP((2,1))
    mlp.activation = bob.machine.Activation.LOG
    mlp.weights = [numpy.array([[.3], [-.42]])]
    mlp.biases = [numpy.array([-.7])]

    self.assertTrue( (mlinear(i) == mlp(i)).all() )
    os.unlink(MACHINE)

  def test05_ComplicatedCorrectness(self):

    # this test is about importing an already create neural network from
    # NeuralLab and trying it with bob clothes. Results generated by bob
    # are verified for correctness using a pre-generated sample.

    m = bob.machine.MLP(bob.io.HDF5File(COMPLICATED))
    data = bob.io.HDF5File(COMPLICATED_OUTPUT)
    for pattern, expected in zip(data.lread("pattern"), data.lread("result")):
      self.assertTrue(abs(m(pattern)[0] - expected) < 1e-8)

    m = bob.machine.MLP(bob.io.HDF5File(COMPLICATED_NOBIAS))
    data = bob.io.HDF5File(COMPLICATED_NOBIAS_OUTPUT)
    for pattern, expected in zip(data.lread("pattern"), data.lread("result")):
      self.assertTrue(abs(m(pattern)[0] - expected) < 1e-8)

  def test05a_ComplicatedCorrectness(self):

    # the same as test05, but with a single pass using the MLP's matrix input

    m = bob.machine.MLP(bob.io.HDF5File(COMPLICATED))
    data = bob.io.HDF5File(COMPLICATED_OUTPUT)
    pat_descr = data.describe('pattern')[0]
    input = numpy.zeros((pat_descr.size, pat_descr.type.shape()[0]), 'float64')
    res_descr = data.describe('result')[0]
    target = numpy.zeros((res_descr.size, res_descr.type.shape()[0]), 'float64')
    for i, (pattern, expected) in enumerate(zip(data.lread("pattern"), data.lread("result"))):
      input[i,:] = pattern
      target[i,:] = expected
    output = m(input)
    self.assertTrue ( (abs(output - target) < 1e-8).all() )

  def test06_Randomization(self):

    # this test makes sure randomization is working as expected on MLPs

    m1 = bob.machine.MLP((2,3,2))
    m1.randomize()

    for k in m1.weights:
      self.assertTrue( (abs(k) <= 0.1).all() )
      self.assertTrue( (k != 0).any() )
    for k in m1.biases:
      self.assertTrue( (abs(k) <= 0.1).all() )
      self.assertTrue( (k != 0).any() )

    for k in range(10): 
      m2 = bob.machine.MLP((2,3,2))
      m2.randomize()
      for w1, w2 in zip(m1.weights, m2.weights):
        self.assertFalse( (w1 == w2).all() )
      for b1, b2 in zip(m1.biases, m2.biases):
        self.assertFalse( (b1 == b2).all() )
      for k in m2.weights:
        self.assertTrue( (abs(k) <= 0.1).all() )
        self.assertTrue( (k != 0).any() )
      for k in m2.biases:
        self.assertTrue( (abs(k) <= 0.1).all() )
        self.assertTrue( (k != 0).any() )

    # we can also reset the margins for randomization
    for k in range(10): 
      m2 = bob.machine.MLP((2,3,2))
      m2.randomize(-0.001, 0.001)
      for w1, w2 in zip(m1.weights, m2.weights):
        self.assertFalse( (w1 == w2).all() )
      for b1, b2 in zip(m1.biases, m2.biases):
        self.assertFalse( (b1 == b2).all() )
      for k in m2.weights:
        self.assertTrue( (abs(k) <= 0.001).all() )
        self.assertTrue( (k != 0).any() )
      for k in m2.biases:
        self.assertTrue( (abs(k) <= 0.001).all() )
        self.assertTrue( (k != 0).any() )
