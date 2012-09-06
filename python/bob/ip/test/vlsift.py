#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Mon Jan 23 20:46:07 2012 +0100
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

"""Tests our SIFT features extractor based on VLFeat
"""

import os, sys
import unittest
import bob
import numpy
import pkg_resources

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def load_image(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join("sift", relative_filename)
  array = bob.io.load(F(filename))
  return array

def equal(x, y, epsilon):
  return (abs(x - y) < epsilon)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class VLSiftTest(unittest.TestCase):
  """Performs various tests"""

  def test01_VLSift_parametrization(self):
    # Creates a VLSIFT object in order to perform parametrization tests
    op = bob.ip.VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 3.)
    self.assertEqual(op.height, 48)
    self.assertEqual(op.width, 64)
    self.assertEqual(op.n_intervals, 3)
    self.assertEqual(op.n_octaves, 5)
    self.assertEqual(op.octave_min, -1)
    self.assertEqual(op.peak_thres, 0.03)
    self.assertEqual(op.edge_thres, 10.)
    self.assertEqual(op.magnif, 3.)
    op.height = 64
    op.width = 96
    op.n_intervals = 4
    op.n_octaves = 6
    op.octave_min = 0
    op.peak_thres = 0.02
    op.edge_thres = 8.
    op.magnif = 2.
    self.assertEqual(op.height, 64)
    self.assertEqual(op.width, 96)
    self.assertEqual(op.n_intervals, 4)
    self.assertEqual(op.n_octaves, 6)
    self.assertEqual(op.octave_min, 0)
    self.assertEqual(op.peak_thres, 0.02)
    self.assertEqual(op.edge_thres, 8.)
    self.assertEqual(op.magnif, 2.)

  def test02_VLSiftKeypointsPython(self):
    # Computes SIFT feature using VLFeat binding
    img = load_image('vlimg_ref.pgm')
    mysift1 = bob.ip.VLSIFT(img.shape[0],img.shape[1], 3, 5, 0)
    # Define keypoints: (y, x, sigma, orientation)
    kp=numpy.array([[75., 50., 1., 1.], [100., 100., 3., 0.]], dtype=numpy.float64)
    # Compute SIFT descriptors at the given keypoints
    out_vl = mysift1(img, kp)
    # Compare to reference
    ref_vl = bob.io.load(F(os.path.join('sift','vlimg_ref_siftKP.hdf5')))
    for kp in range(kp.shape[0]):
      # First 4 values are the keypoint descriptions
      self.assertTrue(equals(out_vl[kp][4:], ref_vl[kp,:], 1e-3))

  def test03_comparison(self):
    # Comparisons tests
    op1 = bob.ip.VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 3.)
    op1b = bob.ip.VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 3.)
    op2 = bob.ip.VLSIFT(48, 64, 3, 5, -1, 0.03, 10., 2.)
    op3 = bob.ip.VLSIFT(48, 64, 3, 5, -1, 0.03, 8., 3.)
    op4 = bob.ip.VLSIFT(48, 64, 3, 5, -1, 0.02, 10., 3.)
    op5 = bob.ip.VLSIFT(48, 64, 3, 5, 0, 0.03, 10., 3.)
    op6 = bob.ip.VLSIFT(48, 64, 3, 4, -1, 0.03, 10., 3.)
    op7 = bob.ip.VLSIFT(48, 64, 2, 5, -1, 0.03, 10., 3.)
    op8 = bob.ip.VLSIFT(48, 96, 3, 5, -1, 0.03, 10., 3.)
    op9 = bob.ip.VLSIFT(128, 64, 3, 5, -1, 0.03, 10., 3.)
    self.assertEqual(op1 == op1, True)
    self.assertEqual(op1 == op1b, True)
    self.assertEqual(op1 == op2, False)
    self.assertEqual(op1 == op3, False)
    self.assertEqual(op1 == op4, False)
    self.assertEqual(op1 == op5, False)
    self.assertEqual(op1 == op6, False)
    self.assertEqual(op1 == op7, False)
    self.assertEqual(op1 == op8, False)
    self.assertEqual(op1 == op9, False)
    self.assertEqual(op1 != op1, False)
    self.assertEqual(op1 != op1b, False)
    self.assertEqual(op1 != op2, True)
    self.assertEqual(op1 != op3, True)
    self.assertEqual(op1 != op4, True)
    self.assertEqual(op1 != op5, True)
    self.assertEqual(op1 != op6, True)
    self.assertEqual(op1 != op7, True)
    self.assertEqual(op1 != op8, True)
    self.assertEqual(op1 != op9, True)
