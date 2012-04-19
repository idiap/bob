#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Apr 19 10:06:07 2012 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

"""Tests our HOG features extractor
"""

import os, sys
import unittest
import bob
import numpy
import math

SRC_A = numpy.array([[0, 0, 4, 0, 0],  [0, 0, 4, 0, 0],  [0, 0, 4, 0, 0],
                     [0, 0, 4, 0, 0],  [0, 0, 4, 0, 0]],  dtype='float64')
MAG1_A = numpy.array([[0, 2, 0, 2, 0], [0, 2, 0, 2, 0], [0, 2, 0, 2, 0],
                      [0, 2, 0, 2, 0], [0, 2, 0, 2, 0]], dtype='float64')
MAG2_A = numpy.array([[0, 4, 0, 4, 0], [0, 4, 0, 4, 0], [0, 4, 0, 4, 0],
                      [0, 4, 0, 4, 0], [0, 4, 0, 4, 0]], dtype='float64')
SQ = math.sqrt(2)
MAGSQRT_A = numpy.array([[0, SQ, 0, SQ, 0], [0, SQ, 0, SQ, 0], [0, SQ, 0, SQ, 0],
                         [0, SQ, 0, SQ, 0], [0, SQ, 0, SQ, 0]], dtype='float64')
PIH = math.pi / 2.
ORI_A = numpy.array([[0, PIH, 0, -PIH, 0], [0, PIH, 0, -PIH, 0], [0, PIH, 0, -PIH, 0],
                     [0, PIH, 0, -PIH, 0], [0, PIH, 0, -PIH, 0]], dtype='float64')
HIST_A = numpy.array([0, 0, 0, 0, 20, 0, 0, 0], dtype='float64')

SRC_B = numpy.array([[0, 0, 0, 0, 0],  [0, 0, 0, 0, 0],  [4, 4, 4, 4, 4],
                     [0, 0, 0, 0, 0],  [0, 0, 0, 0, 0]],  dtype='float64')
MAG_B = numpy.array([[0, 0, 0, 0, 0],  [2, 2, 2, 2, 2],  [0, 0, 0, 0, 0],
                     [2, 2, 2, 2, 2],  [0, 0, 0, 0, 0]],  dtype='float64')
PI = math.pi
ORI_B = numpy.array([[0, 0, 0, 0, 0],  [0, 0, 0, 0, 0],  [0, 0, 0, 0, 0],
                     [PI, PI, PI, PI, PI],  [0, 0, 0, 0, 0]],  dtype='float64')
HIST_B = numpy.array([20, 0, 0, 0, 0, 0, 0, 0], dtype='float64')
EPSILON = 1e-10

HIST_3D = numpy.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], 
                       [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]], dtype='float64')
HIST_NORM_L1 = numpy.zeros(dtype='float64', shape=(20,))

IMG_8x8_A = numpy.array([ [0, 2, 0, 0, 0, 0, 2, 0],
                          [0, 2, 0, 0, 0, 0, 2, 0], 
                          [0, 2, 0, 0, 0, 0, 2, 0], 
                          [0, 2, 0, 0, 0, 0, 2, 0], 
                          [0, 2, 0, 0, 0, 0, 2, 0], 
                          [0, 2, 0, 0, 0, 0, 2, 0], 
                          [0, 2, 0, 0, 0, 0, 2, 0], 
                          [0, 2, 0, 0, 0, 0, 2, 0]], dtype='float64')
HIST_IMG_A = numpy.array([0, 0, 0, 0, 0.5, 0, 0, 0,
                          0, 0, 0, 0, 0.5, 0, 0, 0,
                          0, 0, 0, 0, 0.5, 0, 0, 0,
                          0, 0, 0, 0, 0.5, 0, 0, 0], dtype='float64')

class HOGTest(unittest.TestCase):
  """Performs various tests"""

  def test01_HOGGradientMaps(self):
    """Test the Gradient maps computation"""

    # Declare reference arrays
    hgm = bob.ip.HOGGradientMaps(5,5)
    mag = numpy.zeros(shape=(5,5), dtype='float64')
    ori = numpy.zeros(shape=(5,5), dtype='float64')

    # Magnitude
    hgm.forward(SRC_A, mag, ori)
    numpy.allclose(mag, MAG1_A, EPSILON)
    numpy.allclose(ori, ORI_A, EPSILON)

    # MagnitudeSquare
    hgm.magnitude_type = bob.ip.GradientMagnitudeType.MagnitudeSquare
    hgm.forward(SRC_A, mag, ori)
    numpy.allclose(mag, MAG2_A, EPSILON)
    numpy.allclose(ori, ORI_A, EPSILON)

    # Magnitude
    hgm.magnitude_type = bob.ip.GradientMagnitudeType.SqrtMagnitude
    hgm.forward(SRC_A, mag, ori)
    numpy.allclose(mag, MAGSQRT_A, EPSILON)
    numpy.allclose(ori, ORI_A, EPSILON)

    # Magnitude
    hgm.forward(SRC_B, mag, ori)
    numpy.allclose(mag, MAG_B, EPSILON)
    numpy.allclose(ori, ORI_B, EPSILON) 

  def test02_hogComputeCellHistogram(self):
    """Test the HOG computation for a given cell using hog_compute_cell()"""

    # Check with first input array
    hist = numpy.ndarray(shape=(8,), dtype='float64')
    bob.ip.hog_compute_cell_histogram(MAG1_A, ORI_A, hist)
    numpy.allclose(hist, HIST_A, EPSILON)
    bob.ip.hog_compute_cell_histogram_(MAG1_A, ORI_A, hist)
    numpy.allclose(hist, HIST_A, EPSILON)
    
    # Check with second input array
    bob.ip.hog_compute_cell_histogram(MAG_B, ORI_B, hist)
    numpy.allclose(hist, HIST_B, EPSILON)
    bob.ip.hog_compute_cell_histogram_(MAG_B, ORI_B, hist)
    numpy.allclose(hist, HIST_B, EPSILON)
  
  def test03_hogNormalizeBlock(self):
    """Test the block normalization using hog_normalize_block()"""

    # Vectorizes the 3D histogram into a 1D one
    HIST_1D = numpy.reshape(HIST_3D, (20,))
    # Declares 1D output histogram of size 20
    hist = numpy.ndarray(shape=(20,), dtype='float64')
    # No norm
    bob.ip.hog_normalize_block(HIST_3D, hist, bob.ip.BlockNorm.None)
    numpy.allclose(hist, HIST_1D, EPSILON)
    bob.ip.hog_normalize_block_(HIST_3D, hist, bob.ip.BlockNorm.None)
    numpy.allclose(hist, HIST_1D, EPSILON)
    # L2 Norm
    py_L2ref = HIST_1D / numpy.linalg.norm(HIST_1D)
    bob.ip.hog_normalize_block(HIST_3D, hist)
    numpy.allclose(hist, py_L2ref, EPSILON)
    bob.ip.hog_normalize_block_(HIST_3D, hist)
    numpy.allclose(hist, py_L2ref, EPSILON)
    # L2Hys Norm
    py_L2Hysref = HIST_1D / numpy.linalg.norm(HIST_1D)
    numpy.clip(py_L2Hysref, a_min=0, a_max=0.2) 
    py_L2Hysref = py_L2Hysref / numpy.linalg.norm(py_L2Hysref)
    bob.ip.hog_normalize_block(HIST_3D, hist, bob.ip.BlockNorm.L2Hys)
    numpy.allclose(hist, py_L2Hysref, EPSILON)
    bob.ip.hog_normalize_block_(HIST_3D, hist, bob.ip.BlockNorm.L2Hys)
    numpy.allclose(hist, py_L2Hysref, EPSILON)
    # L1 Norm
    py_L1ref = HIST_1D / numpy.linalg.norm(HIST_1D, 1)
    bob.ip.hog_normalize_block(HIST_3D, hist, bob.ip.BlockNorm.L1) 
    numpy.allclose(hist, py_L1ref, EPSILON)
    bob.ip.hog_normalize_block_(HIST_3D, hist, bob.ip.BlockNorm.L1) 
    numpy.allclose(hist, py_L1ref, EPSILON)
    # L1 Norm sqrt
    py_L1sqrtref = HIST_1D / math.sqrt(numpy.linalg.norm(HIST_1D, 1))
    bob.ip.hog_normalize_block(HIST_3D, hist, bob.ip.BlockNorm.L1sqrt)
    numpy.allclose(hist, py_L1sqrtref, EPSILON)
    bob.ip.hog_normalize_block_(HIST_3D, hist, bob.ip.BlockNorm.L1sqrt)
    numpy.allclose(hist, py_L1sqrtref, EPSILON)

  def test04_HOG(self):
    """Test the HOG class which is used to perform the full feature 
      extraction"""
    
    # HOG features extractor 
    hog = bob.ip.HOG(8,12)
    # Check members
    self.assertTrue( hog.height == 8) 
    self.assertTrue( hog.width == 12)
    self.assertTrue( hog.magnitude_type == bob.ip.GradientMagnitudeType.Magnitude)
    self.assertTrue( hog.n_bins == 8)
    self.assertTrue( hog.full_orientation == False)
    self.assertTrue( hog.cell_y == 4)
    self.assertTrue( hog.cell_x == 4)
    self.assertTrue( hog.cell_ov_y == 0)
    self.assertTrue( hog.cell_ov_x == 0)
    self.assertTrue( hog.block_y == 4)
    self.assertTrue( hog.block_x == 4)
    self.assertTrue( hog.block_ov_y == 0)
    self.assertTrue( hog.block_ov_x == 0)
    self.assertTrue( hog.block_norm == bob.ip.BlockNorm.L2)
    self.assertTrue( hog.block_norm_eps == 1e-10)
    self.assertTrue( hog.block_norm_threshold == 0.2)

    # Resize
    hog.resize(12, 16)
    self.assertTrue( hog.height == 12) 
    self.assertTrue( hog.width == 16)
    
    # Disable block normalization
    hog.disable_block_normalization()
    self.assertTrue( hog.block_y == 1)
    self.assertTrue( hog.block_x == 1)
    self.assertTrue( hog.block_ov_y == 0)
    self.assertTrue( hog.block_ov_x == 0)
    self.assertTrue( hog.block_norm == bob.ip.BlockNorm.None)

    # Get the dimensionality of the output
    self.assertTrue( numpy.array_equal( hog.get_output_shape(), numpy.array([3,4,8]) ))
    hog.resize(16, 16)
    self.assertTrue( numpy.array_equal( hog.get_output_shape(), numpy.array([4,4,8]) ))
    hog.block_y = 4
    hog.block_x = 4
    hog.block_ov_y = 0
    hog.block_ov_x = 0
    self.assertTrue( numpy.array_equal( hog.get_output_shape(), numpy.array([1,1,128]) ))
    hog.n_bins = 12
    hog.block_y = 2
    hog.block_x = 2
    hog.block_ov_y = 1
    hog.block_ov_x = 1
    self.assertTrue( numpy.array_equal( hog.get_output_shape(), numpy.array([3,3,48]) ))

    # Check descriptor computation
    hog.resize(8, 8)
    hog.n_bins = 8
    hog.cell_y = 4
    hog.cell_x = 4
    hog.cell_ov_y = 0
    hog.cell_ov_x = 0
    hog.block_y = 2
    hog.block_x = 2
    hog.block_ov_y = 0
    hog.block_ov_x = 0
    hog.block_norm = bob.ip.BlockNorm.L2
    hist_3D = numpy.ndarray(dtype='float64', shape=(1,1,32))
    hog.forward(IMG_8x8_A, hist_3D)
    self.assertTrue( numpy.allclose( hist_3D, HIST_IMG_A, EPSILON))

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(HOGTest)
