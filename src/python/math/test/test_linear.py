#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Tests Torch Linear Algebra features.
"""

import os, sys
import unittest
import torch

class LinearTest(unittest.TestCase):
  """Tests the Linear Algebra features"""
 
  def test01_eye(self):
    """Tests the eye function"""
    N = 3
    # Matrix to decompose
    A=torch.core.array.float64_2((N,N))
    torch.math.eye(A)

    # Compare to reference
    ref=torch.core.array.float64_2([1,0,0,0,1,0,0,0,1], (N,N))

    self.assertEqual( ((A-ref) < 1e-4).all(), True )


  def test02_diag(self):
    """Tests the diag function"""
    N = 3
    # Matrix to decompose
    d=torch.core.array.float64_1([1,2,3], (N,))
    A=torch.math.diag(d)

    # Compare to reference
    ref=torch.core.array.float64_2([1,0,0,0,2,0,0,0,3], (N,N))

    self.assertEqual( ((A-ref) < 1e-4).all(), True )

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
