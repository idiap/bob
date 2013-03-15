#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri Jan 27 21:06:59 2012 +0100
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

"""Tests bob interior point Linear Programming solvers
"""

import os, sys
import unittest
import bob
import numpy

def generateProblem(n):
  A = numpy.ndarray((n,2*n), numpy.float64)
  b = numpy.ndarray((n,), numpy.float64)
  c = numpy.ndarray((2*n,), numpy.float64)
  x0 = numpy.ndarray((2*n,), numpy.float64)
  sol = numpy.ndarray((n,), numpy.float64)
  A[:] = 0.
  c[:] = 0.
  sol[:] = 0.
  for i in range(n):
    A[i,i] = 1.
    A[i,n+i] = 1.
    for j in range(i+1,n):
      A[j,i] = pow(2., 1+j)
    b[i] = pow(5.,i+1)
    c[i] = -pow(2., n-1-i)
    x0[i] = 1.
  ones = numpy.ndarray((n,), numpy.float64)
  ones[:] = 1.
  A1_1 = numpy.dot(A[:,0:n], ones)
  for i in range(n):
    x0[n+i] = b[i] - A1_1[i]
  sol[n-1] = pow(5.,n)
  return (A,b,c,x0,sol)

class InteriorpointLPTest(unittest.TestCase):
  """Tests the Linear Programming solvers"""
 
  def test01_solvers(self):
    # This test demonstrates how to solve a Linear Programming problem
    # with the provided interior point methods

    eps = 1e-4
    acc = 1e-7
    for N in range(1,10):
      A, b, c, x0, sol = generateProblem(N)

      # short step
      op1 = bob.math.LPInteriorPointShortstep(A.shape[0], A.shape[1], 0.4, acc)
      x = op1.solve(A, b, c, x0)
      # Compare to reference solution
      self.assertEqual( (abs(x-sol) < eps).all(), True )

      # predictor corrector
      op2 = bob.math.LPInteriorPointPredictorCorrector(A.shape[0], A.shape[1], 0.5, 0.25, acc)
      x = op2.solve(A, b, c, x0)
      # Compare to reference solution
      self.assertEqual( (abs(x-sol) < eps).all(), True )

      # long step
      op3 = bob.math.LPInteriorPointLongstep(A.shape[0], A.shape[1], 1e-3, 0.1, acc)
      x = op3.solve(A, b, c, x0)
      # Compare to reference solution
      self.assertEqual( (abs(x-sol) < eps).all(), True )

  def test02_parameters(self):
    op1 = bob.math.LPInteriorPointShortstep(2, 4, 0.4, 1e-6)
    self.assertEqual( op1.m, 2)
    self.assertEqual( op1.n, 4)
    self.assertEqual( op1.theta, 0.4)
    self.assertEqual( op1.epsilon, 1e-6)
    op1b = bob.math.LPInteriorPointShortstep(op1)
    self.assertTrue( op1 == op1b)
    self.assertFalse( op1 != op1b)
    op1b.theta = 0.5
    self.assertFalse( op1 == op1b)
    self.assertTrue( op1 != op1b)
    op1b.reset(3, 6)
    op1b.epsilon = 1e-5
    self.assertEqual( op1b.m, 3)
    self.assertEqual( op1b.n, 6)
    self.assertEqual( op1b.theta, 0.5)
    self.assertEqual( op1b.epsilon, 1e-5)

    op2 = bob.math.LPInteriorPointPredictorCorrector(2, 4, 0.5, 0.25, 1e-6)
    self.assertEqual( op2.m, 2)
    self.assertEqual( op2.n, 4)
    self.assertEqual( op2.theta_pred, 0.5)
    self.assertEqual( op2.theta_corr, 0.25)
    self.assertEqual( op2.epsilon, 1e-6)
    op2b = bob.math.LPInteriorPointPredictorCorrector(op2)
    self.assertTrue( op2 == op2b)
    self.assertFalse( op2 != op2b)
    op2b.theta_pred = 0.4
    self.assertFalse( op2 == op2b)
    self.assertTrue( op2 != op2b)
    op2b.reset(3, 6)
    op2b.theta_corr = 0.2
    op2b.epsilon = 1e-5
    self.assertEqual( op2b.m, 3)
    self.assertEqual( op2b.n, 6)
    self.assertEqual( op2b.theta_pred, 0.4)
    self.assertEqual( op2b.theta_corr, 0.2)
    self.assertEqual( op2b.epsilon, 1e-5)
    op2b.m = 4
    op2b.n = 8
    self.assertEqual( op2b.m, 4)
    self.assertEqual( op2b.n, 8)
 
    op3 = bob.math.LPInteriorPointLongstep(2, 4, 0.4, 0.6, 1e-6)
    self.assertEqual( op3.m, 2)
    self.assertEqual( op3.n, 4)
    self.assertEqual( op3.gamma, 0.4)
    self.assertEqual( op3.sigma, 0.6)
    self.assertEqual( op3.epsilon, 1e-6)
    op3b = bob.math.LPInteriorPointLongstep(op3)
    self.assertTrue( op3 == op3b)
    self.assertFalse( op3 != op3b)
    op3b.gamma = 0.5
    self.assertFalse( op3 == op3b)
    self.assertTrue( op3 != op3b)
    op3b.reset(3, 6)
    op3b.sigma = 0.7
    op3b.epsilon = 1e-5
    self.assertEqual( op3b.m, 3)
    self.assertEqual( op3b.n, 6)
    self.assertEqual( op3b.gamma, 0.5)
    self.assertEqual( op3b.sigma, 0.7)
    self.assertEqual( op3b.epsilon, 1e-5)
    op3b.m = 4
    op3b.n = 8
    self.assertEqual( op3b.m, 4)
    self.assertEqual( op3b.n, 8)
 
  def test03_dual(self):
    A = numpy.array([[1., 0., 1., 0.], [4., 1., 0., 1.]])
    c = numpy.array([-2., -1., 0., 0.])
    op = bob.math.LPInteriorPointShortstep(2, 4, 0.4, 1e-6)
    op.initialize_dual_lambda_mu(A, c)
    lambda_ = op.lambda_
    mu = op.mu
    self.assertTrue(numpy.all( mu >= 0.))

    eps = 1e-4
    At = A.transpose(1,0)
    ref = numpy.dot(At, lambda_) + mu
    self.assertTrue( numpy.all( numpy.fabs( ref - c) <= eps))
