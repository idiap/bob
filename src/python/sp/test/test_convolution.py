#!/usr/bin/env python
#
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# 24 Nov 2010

import os, sys
import unittest
import torch

#############################################################################
# Compare naive DCT/DFT implementation with values returned by Matlab
#############################################################################

########################## Values used for the computation ##################
eps = 1e-3
A10 = torch.core.array.float64_1([0.7094, 0.7547, 0.2760, 0.6797,
  0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404], (10,))
b1_3 = torch.core.array.float64_1([0.5853, 0.2238, 0.7513], (3,))
b1_4 = torch.core.array.float64_1([0.2551, 0.5060, 0.6991, 0.8909], (4,));
b1_5 = torch.core.array.float64_1([0.9593, 0.5472, 0.1386, 0.1493, 0.2575],
  (5,));

RES_A1_10_b1_3_full = torch.core.array.float64_1([0.4152, 0.6005, 0.8634, 
  1.0266, 0.7429, 0.7524, 0.5982, 0.4405, 0.7626, 0.7884, 0.7972, 0.2557], 
  (12,));
RES_A1_10_b1_3_same = torch.core.array.float64_1([0.6005, 0.8634, 1.0266, 
  0.7429, 0.7524, 0.5982, 0.4405, 0.7626, 0.7884, 0.7972], (10,));

RES_A1_10_b1_3_valid = torch.core.array.float64_1([0.8634, 1.0266, 0.7429, 
  0.7524, 0.5982, 0.4405, 0.7626, 0.7884], (8,));

RES_A1_10_b1_4_full = torch.core.array.float64_1([0.1810, 0.5514, 0.9482, 
  1.4726, 1.3763, 1.0940, 1.1761, 0.8846, 0.7250, 1.0268, 1.2872, 1.0930, 
  0.3033], (13,));

RES_A1_10_b1_4_same = torch.core.array.float64_1([0.9482, 1.4726, 1.3763, 
  1.0940, 1.1761, 0.8846, 0.7250, 1.0268, 1.2872, 1.0930], (10,));

RES_A1_10_b1_4_valid = torch.core.array.float64_1([1.4726, 1.3763, 1.0940, 
  1.1761, 0.8846, 0.7250, 1.0268], (7,));

RES_A1_10_b1_5_full = torch.core.array.float64_1([0.6805, 1.1122, 0.7761, 
  1.0136, 1.3340, 0.8442, 0.4665, 0.8386, 1.4028, 0.9804, 0.4244, 0.3188,
  0.2980, 0.0877], (14,));

RES_A1_10_b1_5_same = torch.core.array.float64_1([0.7761, 1.0136, 1.3340, 
  0.8442, 0.4665, 0.8386, 1.4028, 0.9804, 0.4244, 0.3188], (10,));

RES_A1_10_b1_5_valid = torch.core.array.float64_1([1.3340, 0.8442, 0.4665, 
  0.8386, 1.4028, 0.9804], (6,));
#############################################################################

def compare(v1, v2, width):
  return abs(v1-v2) <= width


def test_convolve_1D_nopt(A, b, reference, obj):
  # Compute the convolution product
  res = torch.sp.convolve(A, b)
  
  obj.assertEqual(res.shape(), reference.shape())
  for i in range(res.extent(0)):
    obj.assertTrue(compare(res[i], reference[i], eps))

def test_convolve_1D(A, b, reference, obj, option):
  # Compute the convolution product
  res = torch.sp.convolve(A, b, option)

  obj.assertEqual(res.shape(), reference.shape())
  for i in range(res.extent(0)):
    obj.assertTrue(compare(res[i], reference[i], eps))


##################### Unit Tests ##################  
class ConvolutionTest(unittest.TestCase):
  """Performs convolution product"""

##################### Convolution Tests ##################  
  def test_convolution_1D_10_3_n(self):
    test_convolve_1D_nopt( A10, b1_3, RES_A1_10_b1_3_full, self)

  def test_convolution_1D_10_3_F(self):
    test_convolve_1D( A10, b1_3, RES_A1_10_b1_3_full, self, 0)

  def test_convolution_1D_10_3_S(self):
    test_convolve_1D( A10, b1_3, RES_A1_10_b1_3_same, self, 1)

  def test_convolution_1D_10_3_V(self):
    test_convolve_1D( A10, b1_3, RES_A1_10_b1_3_valid, self, 2)

  def test_convolution_1D_10_4_n(self):
    test_convolve_1D_nopt( A10, b1_4, RES_A1_10_b1_4_full, self)

  def test_convolution_1D_10_4_F(self):
    test_convolve_1D( A10, b1_4, RES_A1_10_b1_4_full, self, 0)

  def test_convolution_1D_10_4_S(self):
    test_convolve_1D( A10, b1_4, RES_A1_10_b1_4_same, self, 1)

  def test_convolution_1D_10_4_V(self):
    test_convolve_1D( A10, b1_4, RES_A1_10_b1_4_valid, self, 2)

  def test_convolution_1D_10_5_n(self):
    test_convolve_1D_nopt( A10, b1_5, RES_A1_10_b1_5_full, self)

  def test_convolution_1D_10_5_F(self):
    test_convolve_1D( A10, b1_5, RES_A1_10_b1_5_full, self, 0)

  def test_convolution_1D_10_5_S(self):
    test_convolve_1D( A10, b1_5, RES_A1_10_b1_5_same, self, 1)

  def test_convolution_1D_10_5_V(self):
    test_convolve_1D( A10, b1_5, RES_A1_10_b1_5_valid, self, 2)


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

