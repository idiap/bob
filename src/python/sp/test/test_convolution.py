#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri Jan 28 21:14:34 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os, sys
import unittest
import bob
import numpy

#############################################################################
# Compare blitz-based convolution product implementation with values returned 
# by Matlab
#############################################################################

########################## Values used for the computation ##################
eps = 1e-3
A10 = numpy.array([0.7094, 0.7547, 0.2760, 0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404], 'float64')
b1_3 = numpy.array([0.5853, 0.2238, 0.7513], 'float64')
b1_4 = numpy.array([0.2551, 0.5060, 0.6991, 0.8909], 'float64')
b1_5 = numpy.array([0.9593, 0.5472, 0.1386, 0.1493, 0.2575], 'float64')

A2_5 = numpy.array([
  0.8407, 0.3500, 0.3517, 0.2858, 0.0759,
  0.2543, 0.1966, 0.8308, 0.7572, 0.0540,
  0.8143, 0.2511, 0.5853, 0.7537, 0.5308,
  0.2435, 0.6160, 0.5497, 0.3804, 0.7792,
  0.9293, 0.4733, 0.9172, 0.5678, 0.9340], 'float64').reshape(5,5)

A4_5 = numpy.array([
  0.8407, 0.3500, 0.3517, 0.2858, 0.0759,
  0.2543, 0.1966, 0.8308, 0.7572, 0.0540,
  0.8143, 0.2511, 0.5853, 0.7537, 0.5308,
  0.2435, 0.6160, 0.5497, 0.3804, 0.7792,
  0.9293, 0.4733, 0.9172, 0.5678, 0.9340], 'float64').reshape(5,5,1,1)

b2_2 = numpy.array([0.1299, 0.4694, 0.5688, 0.0119], 'float64').reshape(2,2)

b2_3 = numpy.array([
  0.3371, 0.3112, 0.6020,
  0.1622, 0.5285, 0.2630,
  0.7943, 0.1656, 0.6541], 'float64').reshape(3,3)


RES_A1_10_b1_3_full = numpy.array([0.4152, 0.6005, 0.8634, 
  1.0266, 0.7429, 0.7524, 0.5982, 0.4405, 0.7626, 0.7884, 0.7972, 0.2557], 
  'float64')
RES_A1_10_b1_3_same = numpy.array([0.6005, 0.8634, 1.0266, 
  0.7429, 0.7524, 0.5982, 0.4405, 0.7626, 0.7884, 0.7972], 'float64')

RES_A1_10_b1_3_valid = numpy.array([0.8634, 1.0266, 0.7429, 
  0.7524, 0.5982, 0.4405, 0.7626, 0.7884], 'float64')

RES_A1_10_b1_4_full = numpy.array([0.1810, 0.5514, 0.9482, 
  1.4726, 1.3763, 1.0940, 1.1761, 0.8846, 0.7250, 1.0268, 1.2872, 1.0930, 
  0.3033], 'float64')

RES_A1_10_b1_4_same = numpy.array([0.9482, 1.4726, 1.3763, 
  1.0940, 1.1761, 0.8846, 0.7250, 1.0268, 1.2872, 1.0930], 'float64');

RES_A1_10_b1_4_valid = numpy.array([1.4726, 1.3763, 1.0940, 
  1.1761, 0.8846, 0.7250, 1.0268], 'float64');

RES_A1_10_b1_5_full = numpy.array([0.6805, 1.1122, 0.7761, 
  1.0136, 1.3340, 0.8442, 0.4665, 0.8386, 1.4028, 0.9804, 0.4244, 0.3188,
  0.2980, 0.0877], 'float64');

RES_A1_10_b1_5_same = numpy.array([0.7761, 1.0136, 1.3340, 
  0.8442, 0.4665, 0.8386, 1.4028, 0.9804, 0.4244, 0.3188], 'float64');

RES_A1_10_b1_5_valid = numpy.array([1.3340, 0.8442, 0.4665, 
  0.8386, 1.4028, 0.9804], 'float64');

RES_A2_5_b2_2_full = numpy.array([
  0.1092, 0.4401, 0.2100, 0.2022, 0.1440, 0.0356,
  0.5113, 0.3540, 0.4044, 0.6551, 0.4090, 0.0262,
  0.2504, 0.5297, 0.6688, 0.8132, 0.4624, 0.2498,
  0.4948, 0.3469, 0.6965, 0.7432, 0.5907, 0.3721,
  0.2592, 0.8510, 0.6613, 0.7272, 0.8356, 0.4477,
  0.5286, 0.2803, 0.5274, 0.3339, 0.5380, 0.0111], 'float64').reshape(6,6)

RES_A2_5_b2_2_same = numpy.array([
  0.3540, 0.4044, 0.6551, 0.4090, 0.0262,
  0.5297, 0.6688, 0.8132, 0.4624, 0.2498,
  0.3469, 0.6965, 0.7432, 0.5907, 0.3721,
  0.8510, 0.6613, 0.7272, 0.8356, 0.4477,
  0.2803, 0.5274, 0.3339, 0.5380, 0.0111], 'float64').reshape(5,5)

RES_A2_5_b2_2_valid = numpy.array([
  0.3540, 0.4044, 0.6551, 0.4090,
  0.5297, 0.6688, 0.8132, 0.4624,
  0.3469, 0.6965, 0.7432, 0.5907,
  0.8510, 0.6613, 0.7272, 0.8356], 'float64').reshape(4,4)

RES_A2_5_b2_3_full = numpy.array([
  0.2834, 0.3796, 0.7336, 0.4165, 0.3262, 0.1957, 0.0457,
  0.2221, 0.6465, 0.9574, 0.9564, 1.0098, 0.5879, 0.0524,
  0.9835, 0.9216, 1.9583, 1.7152, 1.7309, 1.0461, 0.3833,
  0.4161, 0.9528, 1.8242, 2.0354, 2.0621, 1.4545, 0.6439,
  0.9995, 1.0117, 2.5338, 2.1359, 2.4450, 1.7253, 1.1143,
  0.3441, 1.0976, 1.3412, 1.4975, 1.7343, 1.0209, 0.7553,
  0.7381, 0.5299, 1.4147, 0.9125, 1.4358, 0.5261, 0.6109], 'float64').reshape(7,7)

RES_A2_5_b2_3_same = numpy.array([
  0.6465, 0.9574, 0.9564, 1.0098, 0.5879,
  0.9216, 1.9583, 1.7152, 1.7309, 1.0461,
  0.9528, 1.8242, 2.0354, 2.0621, 1.4545,
  1.0117, 2.5338, 2.1359, 2.4450, 1.7253,
  1.0976, 1.3412, 1.4975, 1.7343, 1.0209], 'float64').reshape(5,5)

RES_A2_5_b2_3_valid = numpy.array([
  1.9583, 1.7152, 1.7309,
  1.8242, 2.0354, 2.0621,
  2.5338, 2.1359, 2.4450], 'float64').reshape(3,3)

RES_1 = numpy.array([0.], 'float64');
RES_2 = numpy.array([1.], 'float64');

RES_A2_5_b1_3_0d_full = numpy.array([
  0.492061710000000, 0.204855000000000, 0.205850010000000, 0.167278740000000, 0.044424270000000,
  0.336990450000000, 0.193399980000000, 0.564977700000000, 0.507151200000000, 0.048592620000000,
  1.165140040000000, 0.453922910000000, 0.792741340000000, 0.825323510000000, 0.379786110000000,
  0.515816480000000, 0.564446560000000, 1.076909590000000, 0.960210540000000, 0.615429000000000,
  1.210198180000000, 0.603534720000000, 1.099595910000000, 0.983721670000000, 1.119845200000000,
  0.390918890000000, 0.568725340000000, 0.618258970000000, 0.412868160000000, 0.794442160000000,
  0.698183090000000, 0.355590290000000, 0.689092360000000, 0.426588140000000,
  0.701714200000000], 'float64').reshape(7,5)

RES_A2_5_b1_3_1d_full = numpy.array([
  0.492061710000000, 0.393003660000000, 0.915797920000000, 0.508944200000000, 0.372618520000000, 0.231707960000000, 0.057023670000000,
  0.148841790000000, 0.171982320000000, 0.721321910000000, 0.776827780000000, 0.825247600000000, 0.580969560000000, 0.040570200000000,
  0.476609790000000, 0.329209170000000, 1.010555860000000, 0.760782180000000, 0.919091190000000, 0.685047850000000, 0.398790040000000,
  0.142520550000000, 0.415040100000000, 0.642541760000000, 0.808471780000000, 0.954188890000000, 0.460179480000000, 0.585412960000000,
  0.543919290000000, 0.484999830000000, 1.340944790000000, 0.893192990000000,
  1.362836200000000, 0.635617340000000, 0.701714200000000], 'float64').reshape(5,7)

RES_A4_5_b1_3_0d_full = numpy.array([
  0.492061710000000, 0.204855000000000, 0.205850010000000, 0.167278740000000, 0.044424270000000,
  0.336990450000000, 0.193399980000000, 0.564977700000000, 0.507151200000000, 0.048592620000000,
  1.165140040000000, 0.453922910000000, 0.792741340000000, 0.825323510000000, 0.379786110000000,
  0.515816480000000, 0.564446560000000, 1.076909590000000, 0.960210540000000, 0.615429000000000,
  1.210198180000000, 0.603534720000000, 1.099595910000000, 0.983721670000000, 1.119845200000000,
  0.390918890000000, 0.568725340000000, 0.618258970000000, 0.412868160000000, 0.794442160000000,
  0.698183090000000, 0.355590290000000, 0.689092360000000, 0.426588140000000,
  0.701714200000000], 'float64').reshape(7,5,1,1)
#############################################################################


def compare(v1, v2, width):
  return abs(v1-v2) <= width

def test_convolve_1D_nopt(A, b, res, reference, obj):
  # Compute the convolution product
  res = numpy.zeros( bob.sp.getConvolveOutputSize(A, b), 'float64' )
  bob.sp.convolve(A, b, res)
  
  obj.assertEqual(res.shape, reference.shape)
  for i in range(res.shape[0]):
    obj.assertTrue(compare(res[i], reference[i], eps))

def test_convolve_1D(A, b, res, reference, obj, option):
  # Compute the convolution product
  res = numpy.zeros( bob.sp.getConvolveOutputSize(A, b, option), 'float64' )
  bob.sp.convolve(A, b, res, option)

  obj.assertEqual(res.shape, reference.shape)
  for i in range(res.shape[0]):
    obj.assertTrue(compare(res[i], reference[i], eps))

def test_convolve_2D_nopt(A, b, res, reference, obj):
  # Compute the convolution product
  res = numpy.zeros( bob.sp.getConvolveOutputSize(A, b), 'float64' )
  bob.sp.convolve(A, b, res)
  
  obj.assertEqual(res.shape, reference.shape)
  for i in range(res.shape[0]):
    for j in range(res.shape[1]):
      obj.assertTrue(compare(res[i,j], reference[i,j], eps))

def test_convolve_2D(A, b, res, reference, obj, option):
  # Compute the convolution product
  res = numpy.zeros( bob.sp.getConvolveOutputSize(A, b, option), 'float64' )
  bob.sp.convolve(A, b, res, option)

  obj.assertEqual(res.shape, reference.shape)
  for i in range(res.shape[0]):
    for j in range(res.shape[1]):
      obj.assertTrue(compare(res[i,j], reference[i,j], eps))


##################### Unit Tests ##################  
class ConvolutionTest(unittest.TestCase):
  """Performs convolution product"""

##################### Convolution Tests ##################  
  def test_convolution_1D_10_3_n(self):
    test_convolve_1D_nopt( A10, b1_3, RES_1, RES_A1_10_b1_3_full, self)

  def test_convolution_1D_10_3_F(self):
    test_convolve_1D( A10, b1_3, RES_1, RES_A1_10_b1_3_full, self, 
      bob.sp.ConvolutionSize.Full)

  def test_convolution_1D_10_3_S(self):
    test_convolve_1D( A10, b1_3, RES_1, RES_A1_10_b1_3_same, self, 
      bob.sp.ConvolutionSize.Same)

  def test_convolution_1D_10_3_V(self):
    test_convolve_1D( A10, b1_3, RES_1, RES_A1_10_b1_3_valid, self, 
      bob.sp.ConvolutionSize.Valid)

  def test_convolution_1D_10_4_n(self):
    test_convolve_1D_nopt( A10, b1_4, RES_1, RES_A1_10_b1_4_full, self)

  def test_convolution_1D_10_4_F(self):
    test_convolve_1D( A10, b1_4, RES_1, RES_A1_10_b1_4_full, self, 
      bob.sp.ConvolutionSize.Full)

  def test_convolution_1D_10_4_S(self):
    test_convolve_1D( A10, b1_4, RES_1, RES_A1_10_b1_4_same, self, 
      bob.sp.ConvolutionSize.Same)

  def test_convolution_1D_10_4_V(self):
    test_convolve_1D( A10, b1_4, RES_1, RES_A1_10_b1_4_valid, self, 
      bob.sp.ConvolutionSize.Valid)

  def test_convolution_1D_10_5_n(self):
    test_convolve_1D_nopt( A10, b1_5, RES_1, RES_A1_10_b1_5_full, self)

  def test_convolution_1D_10_5_F(self):
    test_convolve_1D( A10, b1_5, RES_1, RES_A1_10_b1_5_full, self, 
      bob.sp.ConvolutionSize.Full)

  def test_convolution_1D_10_5_S(self):
    test_convolve_1D( A10, b1_5, RES_1, RES_A1_10_b1_5_same, self, 
      bob.sp.ConvolutionSize.Same)

  def test_convolution_1D_10_5_V(self):
    test_convolve_1D( A10, b1_5, RES_1, RES_A1_10_b1_5_valid, self, 
      bob.sp.ConvolutionSize.Valid)

  def test_convolution_2D_5_2_n(self):
    test_convolve_2D_nopt( A2_5, b2_2, RES_2, RES_A2_5_b2_2_full, self)

  def test_convolution_2D_5_2_F(self):
    test_convolve_2D( A2_5, b2_2, RES_2, RES_A2_5_b2_2_full, self, 
      bob.sp.ConvolutionSize.Full)

  def test_convolution_2D_5_2_S(self):
    test_convolve_2D( A2_5, b2_2, RES_2, RES_A2_5_b2_2_same, self, 
      bob.sp.ConvolutionSize.Same)

  def test_convolution_2D_5_2_V(self):
    test_convolve_2D( A2_5, b2_2, RES_2, RES_A2_5_b2_2_valid, self, 
      bob.sp.ConvolutionSize.Valid)

  def test_convolution_2D_5_3_n(self):
    test_convolve_2D_nopt( A2_5, b2_3, RES_2, RES_A2_5_b2_3_full, self)

  def test_convolution_2D_5_3_F(self):
    test_convolve_2D( A2_5, b2_3, RES_2, RES_A2_5_b2_3_full, self, 
      bob.sp.ConvolutionSize.Full)

  def test_convolution_2D_5_3_S(self):
    test_convolve_2D( A2_5, b2_3, RES_2, RES_A2_5_b2_3_same, self, 
      bob.sp.ConvolutionSize.Same)

  def test_convolution_2D_5_3_V(self):
    test_convolve_2D( A2_5, b2_3, RES_2, RES_A2_5_b2_3_valid, self, 
      bob.sp.ConvolutionSize.Valid)

  def test_convolution_2D_sep_0d(self):
    res0 = numpy.zeros((7,5), 'float64')
    bob.sp.convolveSep(A2_5, b1_3, res0, 0, bob.sp.ConvolutionSize.Full)
    self.assertEqual(res0.shape, RES_A2_5_b1_3_0d_full.shape)
    for i in range(res0.shape[0]):
      for j in range(res0.shape[1]):
        self.assertTrue(compare(res0[i,j], RES_A2_5_b1_3_0d_full[i,j], eps))

  def test_convolution_2D_sep_1d(self):
    res1 = numpy.zeros((5,7), 'float64')
    bob.sp.convolveSep(A2_5, b1_3, res1, 1, bob.sp.ConvolutionSize.Full)
    self.assertEqual(res1.shape, RES_A2_5_b1_3_1d_full.shape)
    for i in range(res1.shape[0]):
      for j in range(res1.shape[1]):
        self.assertTrue(compare(res1[i,j], RES_A2_5_b1_3_1d_full[i,j], eps))

  def test_convolution_4D_sep_0d(self):
    res0 = numpy.zeros((7,5,1,1), 'float64')
    bob.sp.convolveSep(A4_5, b1_3, res0, 0, bob.sp.ConvolutionSize.Full)
    self.assertEqual(res0.shape, RES_A4_5_b1_3_0d_full.shape)
    for i in range(res0.shape[0]):
      for j in range(res0.shape[1]):
        for k in range(res0.shape[2]):
          for l in range(res0.shape[3]):
            self.assertTrue(compare(res0[i,j,k,l], RES_A4_5_b1_3_0d_full[i,j,k,l], eps))


##################### Main ##################  
if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()

