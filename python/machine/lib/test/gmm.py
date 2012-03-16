#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
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

"""Tests the GMM machine and the GMMStats container
"""

import os, sys
import unittest
import bob
import numpy
import tempfile

class GMMMachineTest(unittest.TestCase):
  """Performs various GMM machine-related tests."""

  def test01_GMMStats(self):
    """Test a GMMStats"""

    # Initializes a GMMStats
    gs = bob.machine.GMMStats(2,3)
    log_likelihood = -3.
    T = 57
    n = numpy.array([4.37, 5.31], 'float64')
    sumpx = numpy.array([[1., 2., 3.], [4., 5., 6.]], 'float64')
    sumpxx = numpy.array([[10., 20., 30.], [40., 50., 60.]], 'float64')
    gs.log_likelihood = log_likelihood
    gs.T = T
    gs.n = n
    gs.sumPx = sumpx
    gs.sumPxx = sumpxx
    self.assertTrue( gs.log_likelihood == log_likelihood )
    self.assertTrue( gs.T == T )
    self.assertTrue( (gs.n == n).all() )
    self.assertTrue( (gs.sumPx == sumpx).all() )
    self.assertTrue( (gs.sumPxx == sumpxx).all() )

    # Saves and reads from file
    filename = str(tempfile.mkstemp(".hdf5")[1])
    gs.save(bob.io.HDF5File(filename))
    gs_loaded = bob.machine.GMMStats(bob.io.HDF5File(filename))
    self.assertTrue( gs == gs_loaded )
    # Makes them different
    gs_loaded.T = 58
    self.assertFalse( gs == gs_loaded )
    # Accumulates from another GMMStats
    gs2 = bob.machine.GMMStats(2,3)
    gs2.log_likelihood = log_likelihood
    gs2.T = T
    gs2.n = n
    gs2.sumPx = sumpx
    gs2.sumPxx = sumpxx
    gs2 += gs
    eps = 1e-8
    self.assertTrue( gs2.log_likelihood == 2*log_likelihood )
    self.assertTrue( gs2.T == 2*T )
    self.assertTrue( numpy.allclose(gs2.n, 2*n, eps) )
    self.assertTrue( numpy.allclose(gs2.sumPx, 2*sumpx, eps) )
    self.assertTrue( numpy.allclose(gs2.sumPxx, 2*sumpxx, eps) )

    # Reinit and checks for zeros
    gs_loaded.init()
    self.assertTrue( gs_loaded.log_likelihood == 0 )
    self.assertTrue( gs_loaded.T == 0 )
    self.assertTrue( (gs_loaded.n == 0).all() )
    self.assertTrue( (gs_loaded.sumPx == 0).all() )
    self.assertTrue( (gs_loaded.sumPxx == 0).all() )
    # Resize and checks size
    gs_loaded.resize(4,5)
    self.assertTrue( gs_loaded.sumPx.shape[0] == 4) 
    self.assertTrue( gs_loaded.sumPx.shape[1] == 5) 

    # Clean-up
    os.unlink(filename)

  def test02_GMMMachine(self):
    """Test a GMMMachine basic features"""

    weights   = numpy.array([0.5, 0.5], 'float64')
    means     = numpy.array([[3, 70, 0], [4, 72, 0]], 'float64')
    variances = numpy.array([[1, 10, 1], [2, 5, 2]], 'float64')
    varianceThresholds = numpy.array([[0, 0, 0], [0, 0, 0]], 'float64')

    # Initializes a GMMMachine 
    gmm = bob.machine.GMMMachine(2,3)
    # Sets the weights, means, variances and varianceThresholds and
    # Checks correctness
    gmm.weights = weights
    gmm.means = means
    gmm.variances = variances
    gmm.varianceThresholds = varianceThresholds
    self.assertTrue( gmm.DimC == 2 )
    self.assertTrue( gmm.DimD == 3 )
    self.assertTrue( (gmm.weights == weights).all() )
    self.assertTrue( (gmm.means == means).all() )
    self.assertTrue( (gmm.variances == variances).all() )
    self.assertTrue( (gmm.varianceThresholds == varianceThresholds).all() )
   
    # Checks supervector-like accesses
    self.assertTrue( (gmm.meanSupervector == means.reshape(means.size)).all() )
    self.assertTrue( (gmm.varianceSupervector == variances.reshape(variances.size)).all() )
    newMeans = numpy.array([[3, 70, 2], [4, 72, 2]], 'float64')
    newVariances = numpy.array([[1, 1, 1], [2, 2, 2]], 'float64')
    gmm.meanSupervector = newMeans.reshape(newMeans.size)
    gmm.varianceSupervector = newVariances.reshape(newVariances.size)
    self.assertTrue( (gmm.meanSupervector == newMeans.reshape(newMeans.size)).all() )
    self.assertTrue( (gmm.varianceSupervector == newVariances.reshape(newVariances.size)).all() )

    # Checks particular varianceThresholds-related methods
    varianceThresholds1D = numpy.array([0.3, 1, 0.5], 'float64')
    gmm.setVarianceThresholds(varianceThresholds1D)
    self.assertTrue( (gmm.varianceThresholds[0,:] == varianceThresholds1D).all() )
    self.assertTrue( (gmm.varianceThresholds[1,:] == varianceThresholds1D).all() )
    gmm.setVarianceThresholds(0.005)
    self.assertTrue( (gmm.varianceThresholds == 0.005).all() )

    # Checks Gaussians access
    self.assertTrue( (gmm.getGaussian(0).mean == newMeans[0,:]).all() )
    self.assertTrue( (gmm.getGaussian(1).mean == newMeans[1,:]).all() )
    self.assertTrue( (gmm.getGaussian(0).variance == newVariances[0,:]).all() )
    self.assertTrue( (gmm.getGaussian(1).variance == newVariances[1,:]).all() )

    # Checks resize
    gmm.resize(4,5)
    self.assertTrue( gmm.DimC == 4 )
    self.assertTrue( gmm.DimD == 5 )

  def test03_GMMMachine(self):
    """Test a GMMMachine (statistics)"""

    arrayset = bob.io.Arrayset("faithful.torch3_f64.hdf5")
    gmm = bob.machine.GMMMachine(2, 2)
    gmm.weights   = numpy.array([0.5, 0.5], 'float64')
    gmm.means     = numpy.array([[3, 70], [4, 72]], 'float64')
    gmm.variances = numpy.array([[1, 10], [2, 5]], 'float64')
    gmm.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')

    stats = bob.machine.GMMStats(2, 2)
    gmm.accStatistics(arrayset, stats)
    
    stats_ref = bob.machine.GMMStats(bob.io.HDF5File("stats.hdf5"))

    self.assertTrue(stats.T == stats_ref.T)
    self.assertTrue( numpy.allclose(stats.n, stats_ref.n, atol=1e-10) )
    #self.assertTrue( numpy.array_equal(stats.sumPx, stats_ref.sumPx) )
    #Note AA: precision error above
    self.assertTrue ( numpy.allclose(stats.sumPx, stats_ref.sumPx, atol=1e-10) )
    self.assertTrue( numpy.allclose(stats.sumPxx, stats_ref.sumPxx, atol=1e-10) )

  def test04_GMMMachine(self):
    """Test a GMMMachine (log-likelihood computation)"""
    
    data = bob.io.load('data.hdf5')
    gmm = bob.machine.GMMMachine(2, 50)
    gmm.weights   = bob.io.load('weights.hdf5')
    gmm.means     = bob.io.load('means.hdf5')
    gmm.variances = bob.io.load('variances.hdf5')

    # Compare the log-likelihood with the one obtained using Chris Matlab 
    # implementation
    matlab_ll_ref = -2.361583051672024e+02
    self.assertTrue( abs(gmm(data) - matlab_ll_ref) < 1e-10)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(GMMMachineTest)
