#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 20 Jun 2011 15:21:19 CEST 

"""Tests for statistical methods
"""

import os, sys
import unittest
import torch
import numpy

def load_iris():
  """Loads Fisher's Iris Dataset."""
  test_data = '../../measure/test/data/iris.data'
  retval = {'setosa': [], 'versicolor': [], 'virginica': []}
  for line in open(test_data,'rt'):
    if not line.strip(): continue
    s = [k.strip() for k in line.split(',') if line.strip()]
    if s[4].find('setosa') != -1:
      retval['setosa'].append([float(k) for k in s[0:4]])
    elif s[4].find('versicolor') != -1:
      retval['versicolor'].append([float(k) for k in s[0:4]])
    elif s[4].find('virginica') != -1:
      retval['virginica'].append([float(k) for k in s[0:4]])
  for k in retval.keys():
    retval[k] = torch.core.array.array(retval[k], 'float64')
    retval[k].transposeSelf(1,0)
  return retval

class StatsTest(unittest.TestCase):
  """Tests some statistical APIs for Torch"""

  def setUp(self):

    self.data = load_iris()
 
  def test01_scatter(self):

    # This test demonstrates how to use the scatter matrix function of Torch.
    S, M = torch.math.scatter(self.data['setosa'])
    S /= (self.data['setosa'].extent(1)-1)

    # Do the same with numpy and compare. Note that with numpy we are computing
    # the covariance matrix which is the scatter matrix divided by (N-1).
    K = torch.core.array.array(numpy.cov(self.data['setosa'].as_ndarray()))
    self.assertTrue( (abs(S-K) < 1e-10).all() )

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



