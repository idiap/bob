#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <francois.moulin@idiap.ch>

"""Tests machine package
"""

import os, sys
import unittest
import torch

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon)
  
class MachineTest(unittest.TestCase):
  """Performs various machine tests."""

  def test01_Gaussian(self):
    """Test Gaussian"""
    gaussian = torch.machine.Gaussian(2)

    logLH = gaussian.logLikelihood(torch.core.array.array([0.4, 0.2], 'float64'))
    self.assertTrue(equals(logLH, -1.93787706641, 1e-11))
  
  def test02_GMMMachine(self):
    """Test a GMMMachine"""

    arrayset = torch.database.Arrayset("data/faithful.torch3_f64.hdf5")
    gmm = torch.machine.GMMMachine(2, 2)
    gmm.weights   = torch.core.array.array([0.5, 0.5], 'float64')
    gmm.means     = torch.core.array.array([[3, 70], [4, 72]], 'float64')
    gmm.variances = torch.core.array.array([[1, 10], [2, 5]], 'float64')
    gmm.varianceThresholds = torch.core.array.array([[0, 0], [0, 0]], 'float64')

    stats = torch.machine.GMMStats(2, 2)
    gmm.accStatistics(arrayset, stats)
    
    #config = torch.database.HDF5File("data/stats.hdf5")
    #stats.save(config)

    stats_ref = torch.machine.GMMStats(torch.database.HDF5File("data/stats.hdf5"))

    self.assertTrue(stats.T == stats_ref.T)
    self.assertTrue(stats.n == stats_ref.n)
    self.assertTrue(stats.sumPx == stats_ref.sumPx)
    self.assertTrue(stats.sumPxx == stats_ref.sumPxx)

  def test03_EigenMachine(self):
    """Test a EigenMachine"""
    
    m = torch.machine.EigenMachine()
    # Define a machine which projects data along the first and the second dimension
    m.setEigenvaluesvectors(torch.core.array.float64_1([2,1],(2,)), #eigenvalues
       torch.core.array.float64_2([1,0,0,0,1,0],(2,3))) #eigenvectors
    # Keep two output dimensions
    m.setNOutputs(2)
    
    # Define input samples
    s1 = torch.core.array.float64_1([2,1,1],(3,))
    s2 = torch.core.array.float64_1([2,1,37],(3,))
    
    # Define reference array of projected data
    sol1 = torch.core.array.float64_1([2,1],(2,))
    sol2 = torch.core.array.float64_1([0,0],(2,))
    
    # Process and compare to reference solution
    p1 = m.forward(s1)
    self.assertTrue(equals(p1, sol1, 1e-7))
    p2 = m.forward(s2)
    self.assertTrue(equals(p2, sol1, 1e-7))
    
    # Set a mean removal before the projection (required for PCA)
    m.setPreMean( torch.core.array.float64_1([2,1,47],(3,)) )
    
    # Process again and compare to reference solution
    p3 = m.forward(s1)
    self.assertTrue(equals(p3, sol2, 1e-7))
    p4 = m.forward(s2)
    self.assertTrue(equals(p4, sol2, 1e-7))

  """
  def test04_IncompatibleFrameSample(self):
    fs = torch.machine.FrameSample(torch.core.array.float64_1([2,1,1],(3,)))

    gmm = torch.machine.GMMMachine(2,2)
    self.assertRaises(torch.machine.IncompatibleFrameSample, gmm.forward, fs)
    
    kmeans = torch.machine.KMeansMachine(2,2)
    self.assertRaises(torch.machine.IncompatibleFrameSample, kmeans.forward, fs)
  """

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
