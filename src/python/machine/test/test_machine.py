#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <francois.moulin@idiap.ch>

"""Tests machine package
"""

import os, sys
import unittest
import numpy
import torch

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon)
  
class MachineTest(unittest.TestCase):
  """Performs various machine tests."""

  def test01_Gaussian(self):
    """Test Gaussian"""
    gaussian = torch.machine.Gaussian(2)

    logLH = gaussian.logLikelihood(numpy.array([0.4, 0.2], 'float64'))
    self.assertTrue(equals(logLH, -1.93787706641, 1e-11))
  
  def test02_GMMMachine(self):
    """Test a GMMMachine (statistics)"""

    arrayset = torch.io.Arrayset("data/faithful.torch3_f64.hdf5")
    gmm = torch.machine.GMMMachine(2, 2)
    gmm.weights   = numpy.array([0.5, 0.5], 'float64')
    gmm.means     = numpy.array([[3, 70], [4, 72]], 'float64')
    gmm.variances = numpy.array([[1, 10], [2, 5]], 'float64')
    gmm.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')

    stats = torch.machine.GMMStats(2, 2)
    gmm.accStatistics(arrayset, stats)
    
    #config = torch.io.HDF5File("data/stats.hdf5")
    #stats.save(config)

    stats_ref = torch.machine.GMMStats(torch.io.HDF5File("data/stats.hdf5"))

    self.assertTrue(stats.T == stats_ref.T)
    self.assertTrue( numpy.allclose(stats.n, stats_ref.n, atol=1e-10) )
    #self.assertTrue( numpy.array_equal(stats.sumPx, stats_ref.sumPx) )
    #Note AA: precision error above
    self.assertTrue ( numpy.allclose(stats.sumPx, stats_ref.sumPx, atol=1e-10) )
    self.assertTrue( numpy.allclose(stats.sumPxx, stats_ref.sumPxx, atol=1e-10) )

  def test03_GMMMachine(self):
    """Test a GMMMachine (log-likelihood computation)"""
    
    data = torch.io.Array('data/data.hdf5').get()
    gmm = torch.machine.GMMMachine(2, 50)
    gmm.weights   = torch.io.Array('data/weights.hdf5').get()
    gmm.means     = torch.io.Array('data/means.hdf5').get()
    gmm.variances = torch.io.Array('data/variances.hdf5').get()

    # Compare the log-likelihood with the one obtained using Chris Matlab 
    # implementation
    matlab_ll_ref = -2.361583051672024e+02
    self.assertTrue( abs(gmm(data) - matlab_ll_ref) < 1e-4)

  def test04_GMMMachine(self):
    """Test a GMMMachine (Supervectors)"""

    gmm = torch.machine.GMMMachine(2, 3)
    gmm.weights   = numpy.array([0.5, 0.5], 'float64')
    gmm.means     = numpy.array([[3, 70, 5], [4, 72, 14]], 'float64')
    gmm.variances = numpy.array([[1, 10, 9], [2, 5, 5]], 'float64')

    mean_ref = numpy.array([3, 70, 5, 4, 72, 14], 'float64')
    var_ref = numpy.array([1, 10, 9, 2, 5, 5], 'float64')

    array = numpy.ndarray((6,), 'float64')
    # Check mean supervector
    gmm.getMeanSupervector(array)
    self.assertTrue( numpy.array_equal(array, mean_ref) )

    # Check variance supervector
    gmm.getVarianceSupervector(array)
    self.assertTrue( numpy.array_equal(array, var_ref) )

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
