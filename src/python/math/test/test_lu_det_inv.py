#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Tests bob eigenvalue decomposition.
"""

import os, sys
import unittest
import bob
import numpy

class LUDetInvTest(unittest.TestCase):
  """Tests the LU decomposition, determinant computation and matrix inversion"""
 
  def test01_luDecomposition(self):
    # LU decomposition test

    N = 3
    # Matrix to decompose
    A = numpy.array([0.8147, 0.9134, 0.2785, 0.9058, 0.6324, 0.5469, 
                     0.1270, 0.0975, 0.9575], 'float64').reshape(N,N)

    # Scipy reference
    import scipy.linalg
    P, L, U = scipy.linalg.lu(A)
    # bob 
    L1, U1, P1 = bob.math.lu(A)
    self.assertEqual( (abs(P-P1) < 1e-6).all(), True )
    self.assertEqual( (abs(L-L1) < 1e-6).all(), True )
    self.assertEqual( (abs(U-U1) < 1e-6).all(), True )

  def test02_determinant(self):
    # Determinant test

    N = 3
    # Matrix to decompose
    A = numpy.array([0.8147, 0.9134, 0.2785, 0.9058, 0.6324, 0.5469, 
                     0.1270, 0.0975, 0.9575], 'float64').reshape(N,N)

    # numpy reference
    import scipy.linalg
    d = scipy.linalg.det(A)
    # bob
    d1 = bob.math.det(A)
    self.assertEqual( (abs(d-d1) < 1e-6), True)

  def test03_inv(self):
    # Matrix inversion test

    N = 3
    # Matrix to decompose
    A = numpy.array([0.8147, 0.9134, 0.2785, 0.9058, 0.6324, 0.5469, 
                     0.1270, 0.0975, 0.9575], 'float64').reshape(N,N)

    # numpy reference
    import scipy.linalg
    B = scipy.linalg.inv(A)
    # bob
    B1 = bob.math.inv(A)
    self.assertEqual( (abs(B-B1) < 1e-6).all(), True)


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
