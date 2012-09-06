#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Niklas Johansson <niklas.johansson@idiap.ch>
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Apr 6 14:16:13 2011 +0200
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

"""Test the block extractor
"""

import os, sys
import unittest
import bob
import numpy

A_org    = numpy.array(range(1,17), 'float64').reshape((4,4))
A_ans_0_3D  = numpy.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]], 'float64')
A_ans_0_4D  = numpy.array([[[[1, 2], [5, 6]], [[3, 4], [7, 8]]], [[[9, 10], [13, 14]], [[11, 12], [15, 16]]]], 'float64')

class BlockTest(unittest.TestCase):
  """Performs various tests for the zigzag extractor."""
  def test01_block(self):
    shape_4D = (2, 2, 2, 2)
    shape = bob.ip.get_block_4d_output_shape(A_org, 2, 2, 0, 0)
    self.assertTrue( shape == shape_4D)

    B = numpy.ndarray(shape_4D, 'float64')
    bob.ip.block(A_org, B, 2, 2, 0, 0)
    self.assertTrue( (B == A_ans_0_4D).all())
    C = bob.ip.block(A_org, 2, 2, 0, 0)
    self.assertTrue( (C == A_ans_0_4D).all())
    
    shape_3D = (4, 2, 2)
    shape = bob.ip.get_block_3d_output_shape(A_org, 2, 2, 0, 0)
    self.assertTrue( shape == shape_3D)

    B = numpy.ndarray(shape_3D, 'float64')
    bob.ip.block(A_org, B, 2, 2, 0, 0)
    self.assertTrue( (B == A_ans_0_3D).all())
