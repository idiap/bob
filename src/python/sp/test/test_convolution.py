#!/usr/bin/env python
#
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# 28 Jan 2011

import os, sys
import unittest
import torch

#############################################################################
# Compare blitz-based convolution product implementation with values returned 
# by Matlab
#############################################################################

########################## Values used for the computation ##################
eps = 1e-3
A10 = torch.core.array.float64_1([0.7094, 0.7547, 0.2760, 0.6797,
  0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404], (10,))
b1_3 = torch.core.array.float64_1([0.5853, 0.2238, 0.7513], (3,))
b1_4 = torch.core.array.float64_1([0.2551, 0.5060, 0.6991, 0.8909], (4,));
b1_5 = torch.core.array.float64_1([0.9593, 0.5472, 0.1386, 0.1493, 0.2575],
  (5,));

A2_5 = torch.core.array.float64_2([
  0.8407, 0.3500, 0.3517, 0.2858, 0.0759,
  0.2543, 0.1966, 0.8308, 0.7572, 0.0540,
  0.8143, 0.2511, 0.5853, 0.7537, 0.5308,
  0.2435, 0.6160, 0.5497, 0.3804, 0.7792,
  0.9293, 0.4733, 0.9172, 0.5678, 0.9340], (5,5));

b2_2 = torch.core.array.float64_2([
  0.1299, 0.4694,
  0.5688, 0.0119], (2,2));

b2_3 = torch.core.array.float64_2([
  0.3371, 0.3112, 0.6020,
  0.1622, 0.5285, 0.2630,
  0.7943, 0.1656, 0.6541], (3,3));


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

RES_A2_5_b2_2_full = torch.core.array.float64_2([
  0.1092, 0.4401, 0.2100, 0.2022, 0.1440, 0.0356,
  0.5113, 0.3540, 0.4044, 0.6551, 0.4090, 0.0262,
  0.2504, 0.5297, 0.6688, 0.8132, 0.4624, 0.2498,
  0.4948, 0.3469, 0.6965, 0.7432, 0.5907, 0.3721,
  0.2592, 0.8510, 0.6613, 0.7272, 0.8356, 0.4477,
  0.5286, 0.2803, 0.5274, 0.3339, 0.5380, 0.0111], (6,6));

RES_A2_5_b2_2_same = torch.core.array.float64_2([
  0.3540, 0.4044, 0.6551, 0.4090, 0.0262,
  0.5297, 0.6688, 0.8132, 0.4624, 0.2498,
  0.3469, 0.6965, 0.7432, 0.5907, 0.3721,
  0.8510, 0.6613, 0.7272, 0.8356, 0.4477,
  0.2803, 0.5274, 0.3339, 0.5380, 0.0111], (5,5));

RES_A2_5_b2_2_valid = torch.core.array.float64_2([
  0.3540, 0.4044, 0.6551, 0.4090,
  0.5297, 0.6688, 0.8132, 0.4624,
  0.3469, 0.6965, 0.7432, 0.5907,
  0.8510, 0.6613, 0.7272, 0.8356], (4,4));

RES_A2_5_b2_3_full = torch.core.array.float64_2([
  0.2834, 0.3796, 0.7336, 0.4165, 0.3262, 0.1957, 0.0457,
  0.2221, 0.6465, 0.9574, 0.9564, 1.0098, 0.5879, 0.0524,
  0.9835, 0.9216, 1.9583, 1.7152, 1.7309, 1.0461, 0.3833,
  0.4161, 0.9528, 1.8242, 2.0354, 2.0621, 1.4545, 0.6439,
  0.9995, 1.0117, 2.5338, 2.1359, 2.4450, 1.7253, 1.1143,
  0.3441, 1.0976, 1.3412, 1.4975, 1.7343, 1.0209, 0.7553,
  0.7381, 0.5299, 1.4147, 0.9125, 1.4358, 0.5261, 0.6109], (7,7));

RES_A2_5_b2_3_same = torch.core.array.float64_2([
  0.6465, 0.9574, 0.9564, 1.0098, 0.5879,
  0.9216, 1.9583, 1.7152, 1.7309, 1.0461,
  0.9528, 1.8242, 2.0354, 2.0621, 1.4545,
  1.0117, 2.5338, 2.1359, 2.4450, 1.7253,
  1.0976, 1.3412, 1.4975, 1.7343, 1.0209], (5,5));

RES_A2_5_b2_3_valid = torch.core.array.float64_2([
  1.9583, 1.7152, 1.7309,
  1.8242, 2.0354, 2.0621,
  2.5338, 2.1359, 2.4450], (3,3));

RES_1 = torch.core.array.float64_1([0.], (1,));
RES_2 = torch.core.array.float64_2([1.], (1,1));
#############################################################################


def compare(v1, v2, width):
  return abs(v1-v2) <= width

def test_convolve_1D_nopt(A, b, res, reference, obj):
  # Compute the convolution product
  torch.sp.convolve(A, b, res)
  
  obj.assertEqual(res.shape(), reference.shape())
  for i in range(res.extent(0)):
    obj.assertTrue(compare(res[i], reference[i], eps))

def test_convolve_1D(A, b, res, reference, obj, option):
  # Compute the convolution product
  torch.sp.convolve(A, b, res, option)

  obj.assertEqual(res.shape(), reference.shape())
  for i in range(res.extent(0)):
    obj.assertTrue(compare(res[i], reference[i], eps))

def test_convolve_2D_nopt(A, b, res, reference, obj):
  # Compute the convolution product
  torch.sp.convolve(A, b, res)
  
  obj.assertEqual(res.shape(), reference.shape())
  for i in range(res.extent(0)):
    for j in range(res.extent(1)):
      obj.assertTrue(compare(res[i,j], reference[i,j], eps))

def test_convolve_2D(A, b, res, reference, obj, option):
  # Compute the convolution product
  torch.sp.convolve(A, b, res, option)

  obj.assertEqual(res.shape(), reference.shape())
  for i in range(res.extent(0)):
    for j in range(res.extent(1)):
      obj.assertTrue(compare(res[i,j], reference[i,j], eps))


##################### Unit Tests ##################  
class ConvolutionTest(unittest.TestCase):
  """Performs convolution product"""

##################### Convolution Tests ##################  
  def test_convolution_1D_10_3_n(self):
    test_convolve_1D_nopt( A10, b1_3, RES_1, RES_A1_10_b1_3_full, self)

  def test_convolution_1D_10_3_F(self):
    test_convolve_1D( A10, b1_3, RES_1, RES_A1_10_b1_3_full, self, 
      torch.sp.ConvolutionSize.Full)

  def test_convolution_1D_10_3_S(self):
    test_convolve_1D( A10, b1_3, RES_1, RES_A1_10_b1_3_same, self, 
      torch.sp.ConvolutionSize.Same)

  def test_convolution_1D_10_3_V(self):
    test_convolve_1D( A10, b1_3, RES_1, RES_A1_10_b1_3_valid, self, 
      torch.sp.ConvolutionSize.Valid)

  def test_convolution_1D_10_4_n(self):
    test_convolve_1D_nopt( A10, b1_4, RES_1, RES_A1_10_b1_4_full, self)

  def test_convolution_1D_10_4_F(self):
    test_convolve_1D( A10, b1_4, RES_1, RES_A1_10_b1_4_full, self, 
      torch.sp.ConvolutionSize.Full)

  def test_convolution_1D_10_4_S(self):
    test_convolve_1D( A10, b1_4, RES_1, RES_A1_10_b1_4_same, self, 
      torch.sp.ConvolutionSize.Same)

  def test_convolution_1D_10_4_V(self):
    test_convolve_1D( A10, b1_4, RES_1, RES_A1_10_b1_4_valid, self, 
      torch.sp.ConvolutionSize.Valid)

  def test_convolution_1D_10_5_n(self):
    test_convolve_1D_nopt( A10, b1_5, RES_1, RES_A1_10_b1_5_full, self)

  def test_convolution_1D_10_5_F(self):
    test_convolve_1D( A10, b1_5, RES_1, RES_A1_10_b1_5_full, self, 
      torch.sp.ConvolutionSize.Full)

  def test_convolution_1D_10_5_S(self):
    test_convolve_1D( A10, b1_5, RES_1, RES_A1_10_b1_5_same, self, 
      torch.sp.ConvolutionSize.Same)

  def test_convolution_1D_10_5_V(self):
    test_convolve_1D( A10, b1_5, RES_1, RES_A1_10_b1_5_valid, self, 
      torch.sp.ConvolutionSize.Valid)

  def test_convolution_2D_5_2_n(self):
    test_convolve_2D_nopt( A2_5, b2_2, RES_2, RES_A2_5_b2_2_full, self)

  def test_convolution_2D_5_2_F(self):
    test_convolve_2D( A2_5, b2_2, RES_2, RES_A2_5_b2_2_full, self, 
      torch.sp.ConvolutionSize.Full)

  def test_convolution_2D_5_2_S(self):
    test_convolve_2D( A2_5, b2_2, RES_2, RES_A2_5_b2_2_same, self, 
      torch.sp.ConvolutionSize.Same)

  def test_convolution_2D_5_2_V(self):
    test_convolve_2D( A2_5, b2_2, RES_2, RES_A2_5_b2_2_valid, self, 
      torch.sp.ConvolutionSize.Valid)

  def test_convolution_2D_5_3_n(self):
    test_convolve_2D_nopt( A2_5, b2_3, RES_2, RES_A2_5_b2_3_full, self)

  def test_convolution_2D_5_3_F(self):
    test_convolve_2D( A2_5, b2_3, RES_2, RES_A2_5_b2_3_full, self, 
      torch.sp.ConvolutionSize.Full)

  def test_convolution_2D_5_3_S(self):
    test_convolve_2D( A2_5, b2_3, RES_2, RES_A2_5_b2_3_same, self, 
      torch.sp.ConvolutionSize.Same)

  def test_convolution_2D_5_3_V(self):
    test_convolve_2D( A2_5, b2_3, RES_2, RES_A2_5_b2_3_valid, self, 
      torch.sp.ConvolutionSize.Valid)


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

