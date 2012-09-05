#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri Jan 27 13:43:22 2012 +0100
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

"""Tests bob linear solvers A*x=b.
"""

import os, sys
import unittest
import bob
import numpy

class LinsolveTest(unittest.TestCase):
  """Tests the linear solvers"""
 
  def test01_linsolve(self):
    # This test demonstrates how to solve a linear system A*x=b
    # symmetric positive-definite matrix

    N = 3
    # Matrices for the linear system
    A = numpy.array([1., 3., 5., 7., 9., 1., 3., 5., 7.], 'float64').reshape(N,N)
    b = numpy.array([2., 4., 6.], 'float64')

    # Reference solution
    x_ref = numpy.array([3., -2., 1.], 'float64')

    # Matrix for storing the result
    x1 = numpy.ndarray((3,), 'float64')

    # Computes the solution
    bob.math.linsolve(A,x1,b)
    x2 = bob.math.linsolve(A,b)

    # Compare to reference
    self.assertEqual( (abs(x1-x_ref) < 1e-10).all(), True )
    self.assertEqual( (abs(x2-x_ref) < 1e-10).all(), True )

  def test02_linsolveSympos(self):
    # This test demonstrates how to solve a linear system A*x=b
    # when A is a symmetric positive-definite matrix

    N = 3
    # Matrices for the linear system
    A = numpy.array([2., -1., 0., -1, 2., -1., 0., -1., 2.], 'float64').reshape(N,N)
    b = numpy.array([7., 5., 3.], 'float64')

    # Reference solution
    x_ref = numpy.array([8.5, 10., 6.5], 'float64')

    # Matrix for storing the result
    x1 = numpy.ndarray((3,), 'float64')

    # Computes the solution
    bob.math.linsolve_sympos(A,x1,b)
    x2 = bob.math.linsolve_sympos(A,b)

    # Compare to reference
    self.assertEqual( (abs(x1-x_ref) < 1e-10).all(), True )
    self.assertEqual( (abs(x2-x_ref) < 1e-10).all(), True )

  def test03_linsolveCGSympos(self):
    # This test demonstrates how to solve a linear system A*x=b
    # when A is a symmetric positive-definite matrix
    # using a conjugate gradient technique

    N = 3
    # Matrices for the linear system
    A = numpy.array([2., -1., 0., -1, 2., -1., 0., -1., 2.], 'float64').reshape(N,N)
    b = numpy.array([7., 5., 3.], 'float64')

    # Reference solution
    x_ref = numpy.array([8.5, 10., 6.5], 'float64')

    # Matrix for storing the result
    x1 = numpy.ndarray((3,), 'float64')

    # Computes the solution
    eps = 1e-6
    max_iter = 1000
    bob.math.linsolve_cg_sympos(A,x1,b,eps,max_iter)
    x2 = bob.math.linsolve_cg_sympos(A,b,eps,max_iter)

    # Compare to reference
    self.assertEqual( (abs(x1-x_ref) < 2e-6).all(), True )
    self.assertEqual( (abs(x2-x_ref) < 2e-6).all(), True )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(LinsolveTest)
