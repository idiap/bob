#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Tests bob singular value decomposition.
"""

import os, sys
import unittest
import bob
import numpy

class EigTest(unittest.TestCase):
  """Tests the singular alue decomposition based on Lapack"""
 
  def test01_svd(self):
    # This test demonstrates how to compute a singular value decomposition
    # Matrix to decompose
    A = [[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]

    # Do the decomposition using bob (outputs as parameters)
    U1 = numpy.ndarray((3,3), 'float64')
    S1 = numpy.ndarray((3, ), 'float64')
    V1 = numpy.ndarray((3,3), 'float64')
    # Do the decomposition using bob (outputs returned)
    bob.math.svd(A, U1, S1, V1)
    # Do the decomposition using bob (outputs returned)
    U2, S2, V2 = bob.math.svd(A)
    # Do the decomposition using numpy
    Ur, Sr, Vr = numpy.linalg.svd(A, full_matrices=True)
    # Compare
    self.assertEqual( ((S1-Sr) < 1e-10).all(), True )
    self.assertEqual( ((S2-Sr) < 1e-10).all(), True )
    self.assertEqual( ((U1-U2) < 1e-10).all(), True )
    self.assertEqual( ((V1-V2) < 1e-10).all(), True )


  def test02_svd_econ(self):
    # This test demonstrates how to compute a singular value decomposition
    # Matrix to decompose
    A = [[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]
    U = numpy.ndarray((3,3), 'float64')
    S = numpy.ndarray((3, ), 'float64')

    # Do the decomposition using bob
    bob.math.svd(A,U,S)
    # Do the decomposition using numpy
    Ur, Sr, Vr = numpy.linalg.svd(A, full_matrices=False)
    # Compare
    self.assertEqual( ((S-Sr) < 1e-10).all(), True )


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
