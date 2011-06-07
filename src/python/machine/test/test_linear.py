#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 31 May 2011 15:55:54 CEST 

"""Tests on the LinearMachine infrastructure.
"""

import os, sys
import unittest
import math
import torch

MACHINE = 'data/linear-test.hdf5'

class MachineTest(unittest.TestCase):
  """Performs various LinearMachine tests."""

  def test01_Initialization(self):

    # Two inputs and 1 output
    m = torch.machine.LinearMachine(2,1)
    self.assertTrue( (m.weights == 0.0).all() )
    self.assertEqual( m.weights.shape(), (1,2) )
    self.assertTrue( (m.biases == 0.0).all() )
    self.assertEqual( m.biases.shape(), (1,) )

    # Start by providing the data
    w = torch.core.array.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
    b = torch.core.array.array([0.3, -3.0], 'float64')
    m = torch.machine.LinearMachine(w, b)
    isub = torch.core.array.array([0., 0.5, 0.5], 'float64')
    idiv = torch.core.array.array([0.5, 1.0, 1.0], 'float64')
    m.input_subtract = isub
    m.input_divide = idiv
    m.activation = torch.machine.Activation.TANH

    self.assertTrue( (m.input_subtract == isub).all() )
    self.assertTrue( (m.input_divide == idiv).all() )
    self.assertTrue( (m.weights == w).all() )
    self.assertTrue( (m.biases == b). all() )
    self.assertEqual(m.activation, torch.machine.Activation.TANH)
    # Save to file
    # c = torch.config.Configuration()
    # m.save(c)
    # c.save("bla.hdf5")

    # Start by reading data from a file
    c = torch.config.Configuration(MACHINE)
    m = torch.machine.LinearMachine(c)
    self.assertTrue( (m.weights == w).all() )
    self.assertTrue( (m.biases == b). all() )

    # Makes sure we cannot start with incompatible data
    w = torch.core.array.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
    b = torch.core.array.array([0.3, -3.0, 2.7], 'float64') #wrong
    self.assertRaises( torch.machine.NInputsMismatch, 
        torch.machine.LinearMachine, w, b )

  def test02_Correctness(self):

    # Tests the correctness of a linear machine
    c = torch.config.Configuration(MACHINE)
    m = torch.machine.LinearMachine(c)

    def presumed(ivalue):
      """Calculates, by hand, the presumed output given the input"""

      # These are the supposed preloaded values from the file "MACHINE"
      isub = torch.core.array.array([0., 0.5, 0.5], 'float64')
      idiv = torch.core.array.array([0.5, 1.0, 1.0], 'float64')
      w = torch.core.array.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
      b = torch.core.array.array([0.3, -3.0], 'float64')
      act = math.tanh
  
      return torch.core.array.array([ act((w[i,:]*((ivalue-isub)/idiv)).sum() + b[i]) for i in range(w.extent(0)) ], 'float64')

    testing = [
        [1,1,1],
        [0.5,0.2,200],
        [-27,35.77,0],
        [12,0,0],
        ]

    for k in testing:
      input = torch.core.array.array(k, 'float64')
      self.assertTrue ( (abs(presumed(input) - m(input)) < 1e-10).all() )

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
