#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Jul 22 19:34:20 2012 +0200
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

"""Tests the Gaussian
"""

import os, sys
import unittest
import bob
import numpy


eps = 1e-4

class GaussianTest(unittest.TestCase):
  """Performs various tests for the bob.ip.Gaussian class"""

  def test01_parametrization(self):
    # Parametrization tests
    op = bob.ip.Gaussian(1,1,0.5,0.5)
    self.assertEqual(op.radius_y, 1)
    self.assertEqual(op.radius_x, 1)
    self.assertEqual(op.sigma_y, 0.5)
    self.assertEqual(op.sigma_x, 0.5)
    self.assertEqual(op.conv_border, bob.sp.BorderType.Mirror)
    op.radius_y = 2
    op.radius_x = 2
    op.sigma_y = 1.
    op.sigma_x = 1.
    op.conv_border = bob.sp._sp.BorderType.Circular
    self.assertEqual(op.radius_y, 2)
    self.assertEqual(op.radius_x, 2)
    self.assertEqual(op.sigma_y, 1.)
    self.assertEqual(op.sigma_x, 1.)
    self.assertEqual(op.conv_border, bob.sp.BorderType.Circular)
    op.reset(1,1,0.5,0.5,bob.sp.BorderType.Mirror)
    self.assertEqual(op.radius_y, 1)
    self.assertEqual(op.radius_x, 1)
    self.assertEqual(op.sigma_y, 0.5)
    self.assertEqual(op.sigma_x, 0.5)
    self.assertEqual(op.conv_border, bob.sp.BorderType.Mirror)
  
  def test02_processing(self):
    # Processing tests
    op = bob.ip.Gaussian(1,1,0.5,0.5)
    a_uint8 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.uint8)
    a_float64 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.float64)
    a_ref = numpy.array([[1.5325, 2.4260, 3.42602, 4.3195], [5.1065, 6., 7., 7.8935], [8.6805, 9.5740, 10.5740, 11.4675]])
    a_out = numpy.ndarray(dtype=numpy.float64, shape=(3,4))

    op(a_uint8, a_out)
    self.assertEqual(numpy.allclose(a_out, a_ref, eps, eps), True)
    op(a_float64, a_out)
    self.assertEqual(numpy.allclose(a_out, a_ref, eps, eps), True)
    a_out2 = op(a_float64)
    self.assertEqual(numpy.allclose(a_out2, a_ref, eps, eps), True)

  def test03_comparison(self):
    # Comparisons tests
    op1 = bob.ip.Gaussian(1,1,0.5,0.5)
    op1b = bob.ip.Gaussian(1,1,0.5,0.5)
    op2 = bob.ip.Gaussian(1,1,0.5,0.5, bob.sp.BorderType.Circular)
    op3 = bob.ip.Gaussian(1,1,0.5,1.)
    op4 = bob.ip.Gaussian(1,1,1.,0.5)
    op5 = bob.ip.Gaussian(1,2,0.5,0.5)
    op6 = bob.ip.Gaussian(2,1,0.5,0.5)
    self.assertEqual(op1 == op1, True)
    self.assertEqual(op1 == op1b, True)
    self.assertEqual(op1 == op2, False)
    self.assertEqual(op1 == op3, False)
    self.assertEqual(op1 == op4, False)
    self.assertEqual(op1 == op5, False)
    self.assertEqual(op1 == op6, False)
    self.assertEqual(op1 != op1, False)
    self.assertEqual(op1 != op1b, False)
    self.assertEqual(op1 != op2, True)
    self.assertEqual(op1 != op3, True)
    self.assertEqual(op1 != op4, True)
    self.assertEqual(op1 != op5, True)
    self.assertEqual(op1 != op6, True)
