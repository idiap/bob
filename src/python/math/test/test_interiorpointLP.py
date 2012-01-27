#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Tests bob interior point Linear Programming solvers
"""

import os, sys
import unittest
import bob
import numpy

def generateProblem(n):
  A = numpy.ndarray((n,2*n), 'float64')
  b = numpy.ndarray((n,), 'float64')
  c = numpy.ndarray((2*n,), 'float64')
  x0 = numpy.ndarray((2*n,), 'float64')
  sol = numpy.ndarray((n,), 'float64')
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
  ones = numpy.ndarray((n,), 'float64')
  ones[:] = 1.
  A1_1 = numpy.dot(A[:,0:n], ones)
  for i in range(n):
    x0[n+i] = b[i] - A1_1[i]
  sol[n-1] = pow(5.,n)
  return (A,b,c,x0,sol)

class InteriorpointLPTest(unittest.TestCase):
  """Tests the Linear Programming solvers"""
 
  def test01_interiorpointLP(self):
    # This test demonstrates how to solve a Linear Programming problem
    # with the provided interior point methods

    eps = 1e-4
    acc = 1e-7
    for N in range(1,10):
      A, b, c, x0, sol = generateProblem(N)

      # short step
      x = bob.math.interiorpointShortstepLP(A, b, c, 0.4, x0, acc)
      # Compare to reference solution
      self.assertEqual( (abs(x-sol) < eps).all(), True )

      # predictor corrector
      x = bob.math.interiorpointPredictorCorrectorLP(A, b, c, 0.5, 0.25, x0, acc)
      # Compare to reference solution
      self.assertEqual( (abs(x-sol) < eps).all(), True )

      # long step
      x = bob.math.interiorpointLongstepLP(A, b, c, 1e-3, 0.1, x0, acc)
      # Compare to reference solution
      self.assertEqual( (abs(x-sol) < eps).all(), True )
     
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
