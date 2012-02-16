#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Tue May 10 11:35:58 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

"""Tests machine package
"""

import os, sys
import unittest
import numpy
import bob

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon)
  
class MachineTest(unittest.TestCase):
  """Performs various machine tests."""

  def test01_Gaussian(self):
    """Test Gaussian"""
    gaussian = bob.machine.Gaussian(2)

    logLH = gaussian.logLikelihood(numpy.array([0.4, 0.2], 'float64'))
    self.assertTrue(equals(logLH, -1.93787706641, 1e-11))
  
  def test02_GMMMachine(self):
    """Test a GMMMachine (statistics)"""

    arrayset = bob.io.Arrayset("faithful.torch3_f64.hdf5")
    gmm = bob.machine.GMMMachine(2, 2)
    gmm.weights   = numpy.array([0.5, 0.5], 'float64')
    gmm.means     = numpy.array([[3, 70], [4, 72]], 'float64')
    gmm.variances = numpy.array([[1, 10], [2, 5]], 'float64')
    gmm.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')

    stats = bob.machine.GMMStats(2, 2)
    gmm.accStatistics(arrayset, stats)
    
    #config = bob.io.HDF5File("stats.hdf5")
    #stats.save(config)

    stats_ref = bob.machine.GMMStats(bob.io.HDF5File("stats.hdf5"))

    self.assertTrue(stats.T == stats_ref.T)
    self.assertTrue( numpy.allclose(stats.n, stats_ref.n, atol=1e-10) )
    #self.assertTrue( numpy.array_equal(stats.sumPx, stats_ref.sumPx) )
    #Note AA: precision error above
    self.assertTrue ( numpy.allclose(stats.sumPx, stats_ref.sumPx, atol=1e-10) )
    self.assertTrue( numpy.allclose(stats.sumPxx, stats_ref.sumPxx, atol=1e-10) )

  def test03_GMMMachine(self):
    """Test a GMMMachine (log-likelihood computation)"""
    
    data = bob.io.load('data.hdf5')
    gmm = bob.machine.GMMMachine(2, 50)
    gmm.weights   = bob.io.load('weights.hdf5')
    gmm.means     = bob.io.load('means.hdf5')
    gmm.variances = bob.io.load('variances.hdf5')

    # Compare the log-likelihood with the one obtained using Chris Matlab 
    # implementation
    matlab_ll_ref = -2.361583051672024e+02
    self.assertTrue( abs(gmm(data) - matlab_ll_ref) < 1e-4)

  def test04_GMMMachine(self):
    """Test a GMMMachine (Supervectors)"""

    gmm = bob.machine.GMMMachine(2, 3)
    gmm.weights   = numpy.array([0.5, 0.5], 'float64')
    gmm.means     = numpy.array([[3, 70, 5], [4, 72, 14]], 'float64')
    gmm.variances = numpy.array([[1, 10, 9], [2, 5, 5]], 'float64')

    mean_ref = numpy.array([3, 70, 5, 4, 72, 14], 'float64')
    var_ref = numpy.array([1, 10, 9, 2, 5, 5], 'float64')

    # Check mean supervector
    array = gmm.meanSupervector
    self.assertTrue( numpy.array_equal(array, mean_ref) )

    # Check variance supervector
    array = gmm.varianceSupervector
    self.assertTrue( numpy.array_equal(array, var_ref) )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(MachineTest)
