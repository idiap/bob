#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 31 May 2011 15:55:54 CEST 

"""Tests on the LinearMachine infrastructure.
"""

import os, sys
import unittest
import torch

class MachineTest(unittest.TestCase):
  """Performs various LinearMachine tests."""

  def test01_Initialization(self):

    # Two inputs and 1 output
    m = torch.machine.LinearMachine(2,1)
    self.assertTrue( (m.weights == 0.0).all() )
    self.assertEqual( m.weights.shape(), (1,2) )
    self.assertTrue( (m.biases == 0.0).all() )
    self.assertEqual( m.biases.shape(), (1,) )
  
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
