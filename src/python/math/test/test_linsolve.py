#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

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
    self.assertEqual( ((x1-x_ref) < 1e-10).all(), True )
    self.assertEqual( ((x2-x_ref) < 1e-10).all(), True )

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
    bob.math.linsolveSympos(A,x1,b)
    x2 = bob.math.linsolveSympos(A,b)

    # Compare to reference
    self.assertEqual( ((x1-x_ref) < 1e-10).all(), True )
    self.assertEqual( ((x2-x_ref) < 1e-10).all(), True )

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
    bob.math.linsolveCGSympos(A,x1,b,eps,max_iter)
    x2 = bob.math.linsolveCGSympos(A,b,eps,max_iter)

    # Compare to reference
    self.assertEqual( ((x1-x_ref) < 2e-6).all(), True )
    self.assertEqual( ((x2-x_ref) < 2e-6).all(), True )

   
if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()
