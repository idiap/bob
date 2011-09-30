#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Tests Torch Linear Algebra features.
"""

import os, sys
import unittest
import torch
import numpy

class LinearTest(unittest.TestCase):
  """Tests the Linear Algebra features"""
 
  def test01_eye(self):
    """Tests the eye function"""
    N = 3
    # Matrix to decompose
    A=numpy.zeros((N,N), 'float64')
    torch.math.eye(A)

    # Compare to reference
    ref=numpy.array([1,0,0,0,1,0,0,0,1], 'float64').reshape(N,N)

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
