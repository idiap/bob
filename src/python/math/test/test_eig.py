#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <laurent.el-shafey@idiap.ch>

"""Tests Torch eigenvalue decomposition.
"""

import os, sys
import unittest
import torch

class EigTest(unittest.TestCase):
  """Tests the eigenvalue decomposition based on Lapack"""
 
  def test01_eigSymReal(self):
    # This test demonstrates how to compute an eigenvalue decomposition

    N = 3
    # Matrix to decompose
    A=torch.core.array.float64_2([1,2,3,2,4,5,3,5,6],(3,3))

    # Matrix/vector for storing the eigenvectors/values
    V=torch.core.array.float64_2((3,3))
    D=torch.core.array.float64_1((3,))

    # Do the decomposition
    torch.math.eigSymReal(A,V,D)

    # Compare eigenvalues to matlab reference
    ref=torch.core.array.float64_1([-0.5157, 0.1709, 11.3448], (3,))

    self.assertEqual( ((D-ref) < 1e-3).all(), True )

    # TODO: check that D*V*D-1=A
   
 
  def test02_eigSymGen(self):
    # This test demonstrates how to solve A*X=lambda*B*X

    N = 3
    # Input matrices to decompose
    A=torch.core.array.float64_2([1,2,3,2,4,5,3,5,6],(3,3))
    B=torch.core.array.float64_2([2,-1,0,-1,2,-1,0,-1,2],(3,3))

    # Matrix/vector for storing the eigenvectors/values
    V=torch.core.array.float64_2((3,3))
    D=torch.core.array.float64_1((3,))

    # Do the decomposition
    torch.math.eig(A,B,V,D)

    # Compare eigenvalues to matlab reference
    ref=torch.core.array.float64_1([-0.2728,0.0510,17.9718], (3,))

    self.assertEqual( ((D-ref) < 1e-3).all(), True )

    # TODO: check eigenvectors 


if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  #os.chdir('data')
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

