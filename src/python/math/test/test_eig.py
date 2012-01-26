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

class EigTest(unittest.TestCase):
  """Tests the eigenvalue decomposition based on Lapack"""
 
  def test01_eigSymReal(self):
    # This test demonstrates how to compute an eigenvalue decomposition

    N = 3
    # Matrix to decompose
    A = [[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]

    # Do the decomposition (1)
    V1, D1 = bob.math.eigSymReal(A)
    # Do the decomposition (2)
    V2 = numpy.ndarray((N,N), 'float64')
    D2 = numpy.ndarray((N,), 'float64')
    bob.math.eigSymReal(A, V2, D2)

    # Compare eigenvalues to matlab reference
    ref=numpy.array([-0.5157, 0.1709, 11.3448], 'float64')
    self.assertEqual( ((D1-ref) < 1e-3).all(), True )
    self.assertEqual( ((D2-ref) < 1e-3).all(), True )

    # check that V*D*V^-1=A
    iV1 = bob.math.inv(V1)
    VD1 = numpy.dot(V1, numpy.diag(D1))
    VDiV1 = numpy.dot(VD1, iV1)
    self.assertEqual( ((A-VDiV1) < 1e-10).all(), True )
    # check that V*D*V^-1=A
    iV2 = bob.math.inv(V2)
    VD2 = numpy.dot(V2, numpy.diag(D2))
    VDiV2 = numpy.dot(VD2, iV2)
    self.assertEqual( ((A-VDiV2) < 1e-10).all(), True )
   
 
  def test02_eigSymGen(self):
    # This test demonstrates how to solve A*X=lambda*B*X

    N = 3
    # Input matrices to decompose
    A = [[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]
    B = [[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]]

    # Do the decomposition (1)
    V1, D1 = bob.math.eigSym(A,B)
    # Do the decomposition (2)
    V2 = numpy.ndarray((N,N), 'float64')
    D2 = numpy.ndarray((N,), 'float64')
    bob.math.eigSym(A, B, V2, D2)

    # Compare eigenvalues to matlab reference
    ref=numpy.array([17.9718,0.510,-0.2728], 'float64')

    self.assertEqual( ((D1-ref) < 1e-3).all(), True )
    self.assertEqual( ((D2-ref) < 1e-3).all(), True )

    # TODO: check eigenvectors 


  def test03_eigGen(self):
    # This test demonstrates how to solve A*X=lambda*B*X

    N = 3
    # Input matrices to decompose
    A = [[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]
    B = [[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]]


    # Do the decomposition (1)
    V1, D1 = bob.math.eig(A, B)
    # Do the decomposition (2)
    V2 = numpy.zeros((3,3), 'float64')
    D2 = numpy.zeros((3,), 'float64')
    bob.math.eig(A, B, V2, D2)

    # Compare eigenvalues to matlab reference
    ref=numpy.array([-0.2728,0.0510,17.9718], 'float64')

    # needs to reorder the eigenvalues
    self.assertEqual( ((numpy.sort(D1)-ref) < 1e-3).all(), True )
    self.assertEqual( ((numpy.sort(D2)-ref) < 1e-3).all(), True )

    # TODO: check eigenvectors 


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
