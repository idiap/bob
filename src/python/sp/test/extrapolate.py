#!/usr/bin/env python
#
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# 27 Sept 2011

import os, sys
import unittest
import torch

#############################################################################
# Tests blitz-based extrapolation implementation with values returned 
#############################################################################

########################## Values used for the computation ##################
eps = 1e-3
a5 = torch.core.array.float64_1([1,2,3,4,5], (5,))
a14_zeros = torch.core.array.float64_1([0,0,0,0,1,2,3,4,5,0,0,0,0,0], (14,))
a14_twos = torch.core.array.float64_1([2,2,2,2,1,2,3,4,5,2,2,2,2,2], (14,))
a14_nearest = torch.core.array.float64_1([1,1,1,1,1,2,3,4,5,5,5,5,5,5], (14,))
a14_circular = torch.core.array.float64_1([2,3,4,5,1,2,3,4,5,1,2,3,4,5], (14,))
a14_mirror = torch.core.array.float64_1([4,3,2,1,1,2,3,4,5,5,4,3,2,1], (14,))
#############################################################################


def compare(v1, v2, width):
  return abs(v1-v2) <= width

def test_extrapolate_1D(res, reference, obj):
  # Tests the extrapolation
  obj.assertEqual(res.shape(), reference.shape())
  for i in range(res.extent(0)):
    obj.assertTrue(compare(res[i], reference[i], eps))

##################### Unit Tests ##################  
class ExtrapolationTest(unittest.TestCase):
  """Performs extrapolation product"""

##################### Convolution Tests ##################  
  def test_convolution_1D_zeros(self):
    b = torch.core.array.float64_1((14,))
    torch.sp.extrapolateZero(a5,b)
    test_extrapolate_1D(b,a14_zeros,self)

  def test_convolution_1D_twos(self):
    b = torch.core.array.float64_1((14,))
    torch.sp.extrapolateConstant(a5,b,2.)
    test_extrapolate_1D(b,a14_twos,self)

  def test_convolution_1D_nearest(self):
    b = torch.core.array.float64_1((14,))
    torch.sp.extrapolateNearest(a5,b)
    test_extrapolate_1D(b,a14_nearest,self)

  def test_convolution_1D_circular(self):
    b = torch.core.array.float64_1((14,))
    torch.sp.extrapolateCircular(a5,b)
    test_extrapolate_1D(b,a14_circular,self)

  def test_convolution_1D_mirror(self):
    b = torch.core.array.float64_1((14,))
    torch.sp.extrapolateMirror(a5,b)
    test_extrapolate_1D(b,a14_mirror,self)

##################### Main ##################  
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

