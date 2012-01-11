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

    # Do the decomposition
    V, D = bob.math.eigSymReal(A)

    # Compare eigenvalues to matlab reference
    ref=numpy.array([-0.5157, 0.1709, 11.3448], 'float64')

    self.assertEqual( ((D-ref) < 1e-3).all(), True )

    # TODO: check that D*V*D-1=A
   
 
  def test02_eigSymGen(self):
    # This test demonstrates how to solve A*X=lambda*B*X

    N = 3
    # Input matrices to decompose
    A = [[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]
    B = [[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]]

    # Do the decomposition
    V, D = bob.math.eigSym(A,B)

    # Compare eigenvalues to matlab reference
    ref=numpy.array([17.9718,0.510,-0.2728], 'float64')

    self.assertEqual( ((D-ref) < 1e-3).all(), True )

    # TODO: check eigenvectors 


  def test03_eigGen(self):
    # This test demonstrates how to solve A*X=lambda*B*X

    N = 3
    # Input matrices to decompose
    A = [[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]
    B = [[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]]

    # Matrix/vector for storing the eigenvectors/values
    V = numpy.zeros((3,3), 'float64')
    D = numpy.zeros((3,), 'float64')

    # Do the decomposition
    bob.math.eig(A, B, V, D)

    # Compare eigenvalues to matlab reference
    ref=numpy.array([-0.2728,0.0510,17.9718], 'float64')

    # TODO: needs to reorder the eigenvalues
    # self.assertEqual( ((D-ref) < 1e-3).all(), True )

    # TODO: check eigenvectors 


if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  #os.chdir('data')
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()
