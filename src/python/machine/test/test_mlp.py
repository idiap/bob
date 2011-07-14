#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu  7 Jul 17:30:28 2011 

"""Tests on the MLP infrastructure.
"""

import os, sys
import unittest
import math
import torch

MACHINE = 'data/mlp-test.hdf5'
COMPLICATED = 'data/mlp-big.hdf5'
COMPLICATED_OUTPUT = 'data/network.hdf5'
COMPLICATED_NOBIAS = 'data/mlp-big-nobias.hdf5'
COMPLICATED_NOBIAS_OUTPUT = 'data/network-without-bias.hdf5'

class MLPTest(unittest.TestCase):
  """Performs various MLP tests."""

  def test01_Initialization(self):

    # Two inputs and 1 output
    m = torch.machine.MLP((2,1))
    self.assertEqual(m.shape, (2,1))
    self.assertEqual(m.input_divide.extent(0), 2)
    self.assertEqual(m.input_subtract.extent(0), 2)
    self.assertEqual(len(m.weights), 1)
    self.assertEqual(m.weights[0].shape(), (2,1))
    self.assertTrue((m.weights[0] == 0.0).all())
    self.assertEqual(len(m.biases), 1)
    self.assertEqual(m.biases[0].shape(), (1,))
    self.assertTrue((m.biases[0] == 0.0).all())
    self.assertEqual(m.activation, torch.machine.Activation.TANH)

    # 1 hidden layer
    m = torch.machine.MLP((2,3,1))
    self.assertEqual(m.shape, (2,3,1))
    self.assertEqual(m.input_divide.extent(0), 2)
    self.assertEqual(m.input_subtract.extent(0), 2)
    self.assertEqual(len(m.weights), 2)
    self.assertEqual(m.weights[0].shape(), (2,3))
    self.assertTrue((m.weights[0] == 0.0).all())
    self.assertEqual(m.weights[1].shape(), (3,1))
    self.assertTrue((m.weights[1] == 0.0).all())
    self.assertEqual(len(m.biases), 2)
    self.assertEqual(m.biases[0].shape(), (3,))
    self.assertTrue((m.biases[0] == 0.0).all())
    self.assertEqual(m.biases[1].shape(), (1,))
    self.assertTrue((m.biases[1] == 0.0).all())
    self.assertEqual(m.activation, torch.machine.Activation.TANH)

    # 2+ hidden layers, different activation
    m = torch.machine.MLP((2,3,5,1))
    m.activation = torch.machine.Activation.LOG
    self.assertEqual(m.shape, (2,3,5,1))
    self.assertEqual(m.input_divide.extent(0), 2)
    self.assertEqual(m.input_subtract.extent(0), 2)
    self.assertEqual(len(m.weights), 3)
    self.assertEqual(m.weights[0].shape(), (2,3))
    self.assertTrue((m.weights[0] == 0.0).all())
    self.assertEqual(m.weights[1].shape(), (3,5))
    self.assertTrue((m.weights[1] == 0.0).all())
    self.assertEqual(m.weights[2].shape(), (5,1))
    self.assertTrue((m.weights[2] == 0.0).all())
    self.assertEqual(len(m.biases), 3)
    self.assertEqual(m.biases[0].shape(), (3,))
    self.assertTrue((m.biases[0] == 0.0).all())
    self.assertEqual(m.biases[1].shape(), (5,))
    self.assertTrue((m.biases[1] == 0.0).all())
    self.assertEqual(m.biases[2].shape(), (1,))
    self.assertTrue((m.biases[2] == 0.0).all())
    self.assertEqual(m.activation, torch.machine.Activation.LOG)

    # A resize should make the last machine look, structurally,
    # like the first again
    m.shape = (2,1)
    self.assertEqual(m.shape, (2,1))
    self.assertEqual(m.input_divide.extent(0), 2)
    self.assertEqual(m.input_subtract.extent(0), 2)
    self.assertEqual(len(m.weights), 1)
    self.assertEqual(m.weights[0].shape(), (2,1))
    self.assertEqual(len(m.biases), 1)
    self.assertEqual(m.biases[0].shape(), (1,))
    self.assertEqual(m.activation, torch.machine.Activation.LOG)

  def test02_Checks(self):

    # tests if MLPs check wrong settings
    m = torch.machine.MLP((2,1))

    # the MLP shape cannot have a single entry
    self.assertRaises(torch.machine.InvalidShape, 
        setattr, m, 'shape', (5,))

    # you cannot set the weights vector with the wrong size
    self.assertRaises(torch.machine.WeightShapeMismatch,
        setattr, m, 'weights', [torch.core.array.float64_2((3,1))])

    # the same for the bias
    self.assertRaises(torch.machine.BiasShapeMismatch,
        setattr, m, 'biases', [torch.core.array.float64_1((5,))])
    
    # it works though if the sizes are correct
    new_weights = [torch.core.array.float64_2((2,1))]
    new_weights[0].fill(3.14)
    m.weights = new_weights
    self.assertEqual(len(m.weights), 1)
    self.assertTrue( (m.weights[0] == new_weights[0]).all() )

    new_biases = [torch.core.array.float64_1((1,))]
    new_biases[0].fill(5.71)
    m.biases = new_biases
    self.assertEqual(len(m.biases), 1)
    self.assertTrue( (m.biases[0] == new_biases[0]).all() )

  def test03_LoadingAndSaving(self):

    # make shure we can save an load an MLP machine
    weights = []
    weights.append(torch.core.array.array([[.2, -.1, .2], [.2, .3, .9]]))
    weights.append(torch.core.array.array([[.1, .5], [-.1, .2], [-.1, 1.1]])) 
    biases = []
    biases.append(torch.core.array.array([-.1, .3, .1]))
    biases.append(torch.core.array.array([.2, -.1]))
    
    m = torch.machine.MLP((2,3,2))
    m.weights = weights
    m.biases = biases

    m.save(torch.io.HDF5File(MACHINE))
    m2 = torch.machine.MLP(torch.io.HDF5File(MACHINE))
    
    self.assertEqual(m.shape, m2.shape)
    self.assertTrue((m.input_subtract == m2.input_subtract).all())
    self.assertTrue((m.input_divide == m2.input_divide).all())
    for i in range(len(m.weights)):
      self.assertTrue((m.weights[i] == m2.weights[i]).all())
      self.assertTrue((m.biases[i] == m2.biases[i]).all())

  def test04_Correctness(self):

    # makes sure the outputs of the MLP are correct
    m = torch.machine.MLP(torch.io.HDF5File(MACHINE))
    i = torch.core.array.array([.1, .7])
    y = m(i)
    y_exp = torch.core.array.array([0.09596993, 0.6175601])
    self.assertTrue( (abs(y - y_exp) < 1e-6).all() )

    # compares a simple (logistic activation, 1 layer) MLP with a LinearMachine
    mlinear = torch.machine.LinearMachine(2,1)
    mlinear.activation = torch.machine.Activation.LOG
    mlinear.weights = torch.core.array.array([[.3], [-.42]])
    mlinear.biases = torch.core.array.array([-.7])

    mlp = torch.machine.MLP((2,1))
    mlp.activation = torch.machine.Activation.LOG
    mlp.weights = [torch.core.array.array([[.3], [-.42]])]
    mlp.biases = [torch.core.array.array([-.7])]

    self.assertTrue( (mlinear(i) == mlp(i)).all() )

  def test05_ComplicatedCorrectness(self):

    # this test is about importing an already create neural network from
    # NeuralLab and trying it with Torch clothes. Results generated by Torch
    # are verified for correctness using a pre-generated sample.

    m = torch.machine.MLP(torch.io.HDF5File(COMPLICATED))
    data = torch.io.HDF5File(COMPLICATED_OUTPUT)
    for pattern, expected in zip(data.read("pattern"), data.read("result")):
      self.assertTrue(abs(m(pattern)[0] - expected) < 1e-8)

    m = torch.machine.MLP(torch.io.HDF5File(COMPLICATED_NOBIAS))
    data = torch.io.HDF5File(COMPLICATED_NOBIAS_OUTPUT)
    for pattern, expected in zip(data.read("pattern"), data.read("result")):
      self.assertTrue(abs(m(pattern)[0] - expected) < 1e-8)

  def test06_Randomization(self):

    # this test makes sure randomization is working as expected on MLPs

    m1 = torch.machine.MLP((2,3,2))
    m1.randomize()

    for k in m1.weights:
      self.assertTrue( (abs(k) <= 0.1).all() )
      self.assertTrue( (k != 0).any() )
    for k in m1.biases:
      self.assertTrue( (abs(k) <= 0.1).all() )
      self.assertTrue( (k != 0).any() )

    for k in range(10): 
      m2 = torch.machine.MLP((2,3,2))
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
      m2 = torch.machine.MLP((2,3,2))
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



if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()


