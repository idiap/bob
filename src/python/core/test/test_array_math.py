#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 06 Sep 2011 09:30:00 CEST 

"""Tests some array mathematical operations
"""

import os, sys
import unittest
import torch
import math

class ArrayMathTest(unittest.TestCase):
  """Performs various tests for math operations array objects."""

  def test01_atan2(self):

    def atan2(x, y):
      X = torch.core.array.array([x])
      Y = torch.core.array.array([y])
      return torch.core.array.atan2(X,Y)[0]

    self.assertEqual( atan2(1.,1.), math.pi/4 )
    self.assertEqual( atan2(-1.,1.), 3*math.pi/4 )
    self.assertEqual( atan2(1.,-1.), -math.pi/4 )
    self.assertEqual( atan2(-1.,-1.), -3*math.pi/4 )

  def test02_nan_to_zero(self):

    x = torch.core.array.array([0.])

    self.assertEqual( (x/0).nan_to_zero(), x )

  def test03_nan_to_num(self):

    x = torch.core.array.array([0.,1.,-1.])
    m = torch.core.array.array([0.,sys.float_info.max,-sys.float_info.max])

    self.assertEqual( (x/0).nan_to_num(), m )

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  #os.chdir(os.path.join('data', 'video'))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

