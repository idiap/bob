#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Tests Torch square root of a matrix computation.
"""

import os, sys
import unittest
import torch

class SqrtmTest(unittest.TestCase):
  """Tests the square root of a matrix computation based on Lapack"""
 
  def test01_sqrtSymReal(self):
    # This test demonstrates how to compute the square root of a reall
    # symmetric positive-definite matrix

    N = 4
    # Matrix to decompose
    A=torch.core.array.float64_2(
        [1, -1, 0, 0,  -1, 2, -1, 0,
         0, -1, 2, -1, 0, 0, -1, 1],(N,N))

    # Matrix for storing the result
    C=torch.core.array.float64_2((4,4))

    # Computes the square root (using the two different python methods)
    torch.math.sqrtSymReal(A,C)
    B=torch.math.sqrtSymReal(A)

    # Compare square root to matlab reference
    ref=torch.core.array.float64_2(
      [ 0.81549316, -0.54489511, -0.16221167, -0.10838638,
       -0.54489510,  1.19817659, -0.49106981, -0.16221167,
       -0.16221167, -0.49106981,  1.19817659, -0.54489511,
       -0.10838638, -0.16221167, -0.54489511,  0.81549316], (4,4))
    self.assertEqual( ((B-ref) < 1e-4).all(), True )
    self.assertEqual( ((C-ref) < 1e-4).all(), True )
   
if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
