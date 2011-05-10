#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <francois.moulin@idiap.ch>

"""Tests machine package
"""

import os, sys
import unittest
import torch

class MachineTest(unittest.TestCase):
  """Performs various machine tests."""
  
  def test01_GMMMachine(self):
    """Test a GMMMachine"""
    
    sampler = torch.trainer.SimpleFrameSampler(torch.database.Arrayset("data/faithful.torch3.bindata"))

    gmm = torch.machine.GMMMachine(2, 2)
    gmm.weights   = torch.core.array.array([0.5, 0.5], 'float64')
    gmm.means     = torch.core.array.array([[3, 70], [4, 72]], 'float64')
    gmm.variances = torch.core.array.array([[1, 10], [2, 5]], 'float64')
    gmm.varianceThresholds = torch.core.array.array([[0, 0], [0, 0]], 'float64')

    gmm.print_()

    stats = torch.machine.GMMStats(2, 2)

    gmm.accStatistics(sampler, stats)
    
    stats.print_()

    #TODO Add asserts



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