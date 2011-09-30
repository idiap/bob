#!/usr/bin/env python

"""Tests on the ZTNorm function
"""

import os, sys
import unittest
import numpy
import torch

def sameValue(vect_A, vect_B):
  sameMatrix = numpy.zeros((vect_A.shape[0], vect_B.shape[0]), 'bool')

  for j in range(vect_A.shape[0]):
    for i in range(vect_B.shape[0]):
      sameMatrix[j, i] = (vect_A[j] == vect_B[i])

  return sameMatrix
  
class ZTNormTest(unittest.TestCase):
  """Performs various ZTNorm tests."""

  def test01_ztnorm_simple(self):
    # 3x5
    my_A = numpy.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 8],[7, 6, 5, 4, 3]],'float64')
    # 3x4
    my_B = numpy.array([[5, 4, 7, 8],[9, 8, 7, 4],[5, 6, 3, 2]],'float64')
    # 2x5
    my_C = numpy.array([[5, 4, 3, 2, 1],[2, 1, 2, 3, 4]],'float64')
    # 2x4
    my_D = numpy.array([[8, 6, 4, 2],[0, 2, 4, 6]],'float64')
    
    # 4x1
    znorm_id = numpy.array([1, 2, 3, 4],'uint32')
    # 2x1
    tnorm_id = numpy.array([1, 5],'uint32')

    scores = torch.machine.ztnorm(my_A, my_B, my_C, my_D, sameValue(tnorm_id, znorm_id))

    ref_scores = numpy.array([[-4.45473107e+00, -3.29289322e+00, -1.50519101e+01, -8.42086557e-01, 6.46544511e-03], [-8.27619927e-01,  7.07106781e-01,  1.13757710e+01,  2.01641412e+00, 7.63765080e-01], [ 2.52913570e+00,  2.70710678e+00,  1.24400233e+01,  7.07106781e-01, 6.46544511e-03]], 'float64')
    
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

  def test02_ztnorm_big(self):
    my_A = torch.io.Array("data/ztnorm_eval_eval.mat").get()
    my_B = torch.io.Array("data/ztnorm_znorm_eval.mat").get()
    my_C = torch.io.Array("data/ztnorm_eval_tnorm.mat").get()
    my_D = torch.io.Array("data/ztnorm_znorm_tnorm.mat").get()

    ref_scores = torch.io.Array("data/ztnorm_result.mat").get()
    scores = torch.machine.ztnorm(my_A, my_B, my_C, my_D)
    
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())
    
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
