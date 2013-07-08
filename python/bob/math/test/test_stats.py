#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon Jun 20 16:15:36 2011 +0200
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

"""Tests for statistical methods
"""

import os, sys
import unittest
import bob
import numpy

def means(data):
  return numpy.mean(data, axis=0)

def scatters(data):
  # Step 1: compute the class means mu_c, starting from the sum_c
  mu_c = numpy.array([numpy.mean(data[k][:], axis=0) for k in range(len(data))])

  # Step 2: computes the number of elements in each class
  n_c = numpy.array([data[k].shape[0] for k in range(len(data))])

  # Step 3: computes the global mean mu
  mu = numpy.sum(mu_c.T * n_c, axis=1) / sum(data[k].shape[0] for k in range(len(data)))

  # Step 4: compute the between-class scatter Sb
  mu_c_mu = (mu_c - mu)
  Sb = numpy.dot(n_c * mu_c_mu.T, mu_c_mu)

  # Step 5: compute the within-class scatter Sw
  Sw = numpy.zeros((data[0].shape[1], data[0].shape[1]), dtype=float)
  for k in range(len(data)):
    X_c_mu_c = (data[k][:] - mu_c[k,:])
    Sw += numpy.dot(X_c_mu_c.T, X_c_mu_c)

  return (Sw, Sb, mu)


class StatsTest(unittest.TestCase):
  """Tests some statistical APIs for bob"""

  def setUp(self):

    self.data = bob.db.iris.data().values()
    self.data_c0 = numpy.vstack(bob.db.iris.data()['setosa'])
 
  def test01_scatter(self):

    # This test demonstrates how to use the scatter matrix function of bob.
    S, M = bob.math.scatter(self.data_c0.T)
    S /= (self.data_c0.shape[1]-1)

    # Do the same with numpy and compare. Note that with numpy we are computing
    # the covariance matrix which is the scatter matrix divided by (N-1).
    K = numpy.array(numpy.cov(self.data_c0))
    M_ = means(self.data_c0.T)
    self.assertTrue( (abs(S-K) < 1e-10).all() )
    self.assertTrue( (abs(M-M_) < 1e-10).all() )

  def test02_scatters(self):
    
    # Compares bob's implementation against pythonic one
    # 1. python
    Sw_, Sb_, m_ = scatters(self.data)

    # 2.a. bob
    Sw, Sb, m = bob.math.scatters(self.data)
    # 3.a. comparison
    self.assertTrue(numpy.allclose(Sw, Sw_) )
    self.assertTrue(numpy.allclose(Sb, Sb_) )
    self.assertTrue(numpy.allclose(m, m_) )

    N = self.data[0].shape[1]
    # 2.b. bob
    Sw = numpy.ndarray((N,N), numpy.float64)
    Sb = numpy.ndarray((N,N), numpy.float64)
    m = numpy.ndarray((N,), numpy.float64)
    bob.math.scatters(self.data, Sw, Sb, m)
    # 3.b comparison
    self.assertTrue(numpy.allclose(Sw, Sw_) )
    self.assertTrue(numpy.allclose(Sb, Sb_) )
    self.assertTrue(numpy.allclose(m, m_) )

    # 2.c. bob
    Sw = numpy.ndarray((N,N), numpy.float64)
    Sb = numpy.ndarray((N,N), numpy.float64)
    bob.math.scatters(self.data, Sw, Sb)
    # 3.c comparison
    self.assertTrue(numpy.allclose(Sw, Sw_) )
    self.assertTrue(numpy.allclose(Sb, Sb_) )
