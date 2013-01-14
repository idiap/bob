#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Dec 9 16:32:00 2012 +0100
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

"""Tests for the PAVA-like algorithm
"""

import os, sys
import unittest
import bob
import numpy

class PavxTest(unittest.TestCase):
  """Tests the pavx function of bob"""

  def test01_pavx(self):

    # Reference obtained using bosaris toolkit 1.06
    # Sample 1
    y1 = numpy.array([ 58.4666,  67.1040,  73.1806,  77.0896,  85.8816,
                       89.6381, 101.6651, 102.5587, 109.7933, 117.5715, 
                      118.1671, 138.3151, 141.9755, 145.7352, 159.1108,
                      156.8654, 168.6932, 175.2756])
    ghat1_ref = numpy.array([ 58.4666,  67.1040,  73.1806,  77.0896,  85.8816,
                              89.6381, 101.6651, 102.5587, 109.7933, 117.5715,
                             118.1671, 138.3151, 141.9755, 145.7352, 157.9881,
                             157.9881, 168.6932, 175.2756])
    w1_ref = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1])
    h1_ref = numpy.array([ 58.4666,  67.1040,  73.1806,  77.0896,  85.8816,
                           89.6381, 101.6651, 102.5587, 109.7933, 117.5715,
                          118.1671, 138.3151, 141.9755, 145.7352, 157.9881,
                          168.6932, 175.2756])
    # Sample 2
    y2 = numpy.array([ 46.1093,  64.3255,  76.5252,  89.0061, 100.4421,
                       92.8593,  84.0840,  98.5769, 102.3841, 143.5045,
                      120.8439, 141.4807, 139.0758, 156.8861, 147.3515,
                      147.9773, 154.7762, 180.8819])
    ghat2_ref = numpy.array([ 46.1093,  64.3255,  76.5252,  89.0061,  92.4618,
                              92.4618,  92.4618,  98.5769, 102.3841, 132.1742,
                             132.1742, 140.2783, 140.2783, 150.7383, 150.7383,
                             150.7383, 154.7762, 180.8819])
    w2_ref = numpy.array([1, 1, 1, 1, 3, 1, 1, 2, 2, 3, 1, 1])
    h2_ref = numpy.array([ 46.1093,  64.3255,  76.5252,  89.0061,  92.4618,  
                           98.5769, 102.3841, 132.1742, 140.2783, 150.7383,
                          154.7762,  180.8819])

    # Make a full test for a given sample
    def pavx_internal_test(self, y, ghat_ref, w_ref, h_ref):
      ghat = numpy.ndarray(dtype=numpy.float64, shape=y.shape)
      bob.math.pavx(y, ghat)
      self.assertTrue( numpy.all(numpy.abs(ghat - ghat_ref) < 1e-4), True)
      bob.math.pavx_(y, ghat)
      self.assertTrue( numpy.all(numpy.abs(ghat - ghat_ref) < 1e-4), True)
      w=bob.math.pavxWidth(y, ghat)
      self.assertTrue( numpy.all(numpy.abs(ghat - ghat_ref) < 1e-4), True)
      self.assertTrue( numpy.all(numpy.abs(w - w_ref) < 1e-4), True)
      ret=bob.math.pavxWidthHeight(y, ghat)
      self.assertTrue( numpy.all(numpy.abs(ghat - ghat_ref) < 1e-4), True)
      self.assertTrue( numpy.all(numpy.abs(ret[0] - w_ref) < 1e-4), True)
      self.assertTrue( numpy.all(numpy.abs(ret[1] - h_ref) < 1e-4), True)

    # Test using sample 1
    pavx_internal_test(self, y1, ghat1_ref, w1_ref, h1_ref)
    # Test using sample 2
    pavx_internal_test(self, y2, ghat2_ref, w2_ref, h2_ref)

