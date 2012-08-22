#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Feb 15 23:24:35 2012 +0200
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

"""Tests on the JFAMachine infrastructure.
"""

import os, sys
import unittest
import math
import bob
import numpy
import tempfile

class JFAMachineTest(unittest.TestCase):
  """Performs various JFAMachine tests."""
  
  def test01_JFABaseMachine(self):

    # Creates a UBM
    weights = numpy.array([0.4, 0.6], 'float64')
    means = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
    variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64') 
    ubm = bob.machine.GMMMachine(2,3)
    ubm.weights = weights
    ubm.means = means
    ubm.variances = variances

    # Creates a JFABaseMachine
    U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 'float64')
    V = numpy.array([[6, 5], [4, 3], [2, 1], [1, 2], [3, 4], [5, 6]], 'float64')
    d = numpy.array([0, 1, 0, 1, 0, 1], 'float64')
    m = bob.machine.JFABaseMachine(ubm)
    self.assertTrue( m.dim_ru == 1)
    self.assertTrue( m.dim_rv == 1)

    # Checks for correctness
    m.resize(2,2) 
    m.u = U
    m.v = V
    m.d = d 
    self.assertTrue( (m.u == U).all() )
    self.assertTrue( (m.v == V).all() )
    self.assertTrue( (m.d == d).all() )
    self.assertTrue( m.dim_c == 2)
    self.assertTrue( m.dim_d == 3)
    self.assertTrue( m.dim_cd == 6)
    self.assertTrue( m.dim_ru == 2)
    self.assertTrue( m.dim_rv == 2)
   
    # Saves and loads
    filename = str(tempfile.mkstemp(".hdf5")[1])
    m.save(bob.io.HDF5File(filename, 'w'))
    m_loaded = bob.machine.JFABaseMachine(bob.io.HDF5File(filename))
    m_loaded.ubm = ubm
    self.assertTrue( m == m_loaded )

    # Clean-up
    os.unlink(filename)

  def test02_JFAMachine(self):

    # Creates a UBM
    weights = numpy.array([0.4, 0.6], 'float64')
    means = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
    variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64') 
    ubm = bob.machine.GMMMachine(2,3)
    ubm.weights = weights
    ubm.means = means
    ubm.variances = variances

    # Creates a JFABaseMachine
    U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 'float64')
    V = numpy.array([[6, 5], [4, 3], [2, 1], [1, 2], [3, 4], [5, 6]], 'float64')
    d = numpy.array([0, 1, 0, 1, 0, 1], 'float64')
    base = bob.machine.JFABaseMachine(ubm,2,2)
    base.u = U
    base.v = V
    base.d = d 

    # Creates a JFAMachine
    y = numpy.array([1,2], 'float64')
    z = numpy.array([3,4,1,2,0,1], 'float64')
    m = bob.machine.JFAMachine(base)
    m.y = y
    m.z = z
    self.assertTrue( m.dim_c == 2)
    self.assertTrue( m.dim_d == 3)
    self.assertTrue( m.dim_cd == 6)
    self.assertTrue( m.dim_ru == 2)
    self.assertTrue( m.dim_rv == 2)
    self.assertTrue( (m.y == y).all() )
    self.assertTrue( (m.z == z).all() )

    # Saves and loads
    filename = str(tempfile.mkstemp(".hdf5")[1])
    m.save(bob.io.HDF5File(filename, 'w'))
    m_loaded = bob.machine.JFAMachine(bob.io.HDF5File(filename))
    m_loaded.jfa_base = base
    self.assertTrue( m == m_loaded )

    # Defines GMMStats
    gs = bob.machine.GMMStats(2,3)
    log_likelihood = -3.
    T = 1
    n = numpy.array([0.4, 0.6], 'float64')
    sumpx = numpy.array([[1., 2., 3.], [4., 5., 6.]], 'float64')
    sumpxx = numpy.array([[10., 20., 30.], [40., 50., 60.]], 'float64')
    gs.log_likelihood = log_likelihood
    gs.t = T
    gs.n = n
    gs.sum_px = sumpx
    gs.sum_pxx = sumpxx

    # Forward GMMStats and check estimated value of the x speaker factor
    eps = 1e-10
    x_ref = numpy.array([0.291042849767692, 0.310273618998444], 'float64')
    score_ref = -2.111577181208289
    score = m.forward(gs)
    self.assertTrue( numpy.allclose(m.x, x_ref, eps) )
    self.assertTrue( abs(score_ref-score) < eps )

    # Clean-up
    os.unlink(filename)
  
  def test03_ISVMachine(self):

    # Creates a UBM
    weights = numpy.array([0.4, 0.6], 'float64')
    means = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
    variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64') 
    ubm = bob.machine.GMMMachine(2,3)
    ubm.weights = weights
    ubm.means = means
    ubm.variances = variances

    # Creates a JFABaseMachine
    U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 'float64')
    V = numpy.array([[0], [0], [0], [0], [0], [0]], 'float64')
    d = numpy.array([0, 1, 0, 1, 0, 1], 'float64')
    base = bob.machine.JFABaseMachine(ubm,2)
    base.u = U
    base.v = V
    base.d = d

    # Creates a JFAMachine
    z = numpy.array([3,4,1,2,0,1], 'float64')
    y = numpy.array([0], 'float64')
    m = bob.machine.JFAMachine(base)
    m.y = y
    m.z = z
    self.assertTrue( m.dim_c == 2)
    self.assertTrue( m.dim_d == 3)
    self.assertTrue( m.dim_cd == 6)
    self.assertTrue( m.dim_ru == 2)
    self.assertTrue( m.dim_rv == 1)
    self.assertTrue( (m.z == z).all() )

    # Saves and loads
    filename = str(tempfile.mkstemp(".hdf5")[1])
    m.save(bob.io.HDF5File(filename, 'w'))
    m_loaded = bob.machine.JFAMachine(bob.io.HDF5File(filename))
    m_loaded.jfa_base = base
    self.assertTrue( m == m_loaded )

    # Defines GMMStats
    gs = bob.machine.GMMStats(2,3)
    log_likelihood = -3.
    T = 1
    n = numpy.array([0.4, 0.6], 'float64')
    sumpx = numpy.array([[1., 2., 3.], [4., 5., 6.]], 'float64')
    sumpxx = numpy.array([[10., 20., 30.], [40., 50., 60.]], 'float64')
    gs.log_likelihood = log_likelihood
    gs.t = T
    gs.n = n
    gs.sum_px = sumpx
    gs.sum_pxx = sumpxx

    # Forward GMMStats and check estimated value of the x speaker factor
    eps = 1e-10
    x_ref = numpy.array([0.291042849767692, 0.310273618998444], 'float64')
    score_ref = -3.280498193082100

    score = m.forward(gs)
    self.assertTrue( numpy.allclose(m.x, x_ref, eps) )
    self.assertTrue( abs(score_ref-score) < eps )

    # Clean-up
    os.unlink(filename)


# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(JFAMachineTest)
