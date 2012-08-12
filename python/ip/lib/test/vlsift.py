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

def load_image(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join("sift", relative_filename)
  array = bob.io.load(filename)
  return array

def equal(x, y, epsilon):
  return (abs(x - y) < epsilon)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class VLSiftTest(unittest.TestCase):
  """Performs various tests"""

  def test01_VLSiftKeypointsPython(self):
    # Computes SIFT feature using VLFeat binding
    img = load_image('vlimg_ref.pgm')
    mysift1 = bob.ip.VLSIFT(img.shape[0],img.shape[1], 3, 5, 0)
    # Define keypoints: (y, x, sigma, orientation)
    kp=numpy.array([[75., 50., 1., 1.], [100., 100., 3., 0.]], dtype=numpy.float64)
    # Compute SIFT descriptors at the given keypoints
    out_vl = mysift1(img, kp)
    # Compare to reference
    ref_vl = bob.io.load(os.path.join('sift','vlimg_ref_siftKP.hdf5'))
    for kp in range(kp.shape[0]):
      # First 4 values are the keypoint descriptions
      self.assertTrue(equals(out_vl[kp][4:], ref_vl[kp,:], 1e-3))

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(VLSiftTest)
