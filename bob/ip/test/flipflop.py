#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
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

"""Test the flip and flop operations
"""

import os, sys
import unittest
import bob
import numpy

A_org       = numpy.array(range(1,5), numpy.float64).reshape((2,2))
A_ans_flip  = numpy.array([[3, 4], [1, 2]], numpy.float64)
A_ans_flop  = numpy.array([[2, 1], [4, 3]], numpy.float64)
A3_org      = numpy.array(range(1,13), numpy.float64).reshape((3,2,2))
A3_ans_flip = numpy.array([[[3, 4], [1, 2]], [[7, 8], [5, 6]], [[11, 12], [9,10]]], numpy.float64).reshape((3,2,2))
A3_ans_flop = numpy.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]], [[10, 9], [12,11]]], numpy.float64).reshape((3,2,2))

class FlipFlopTest(unittest.TestCase):
  """Performs various tests for the flip and flop operations."""
  def test01_flip_2D(self):
    B = numpy.ndarray((2,2), numpy.float64)
    bob.ip.flip(A_org, B)
    self.assertTrue( (B == A_ans_flip).all())
    C = bob.ip.flip(A_org)
    self.assertTrue( (C == A_ans_flip).all())

  def test02_flop_2D(self):
    B = numpy.ndarray((2,2), numpy.float64)
    bob.ip.flop(A_org, B)
    self.assertTrue( (B == A_ans_flop).all())
    C = bob.ip.flop(A_org)
    self.assertTrue( (C == A_ans_flop).all())

  def test03_flip_3D(self):
    B = numpy.ndarray((3,2,2), numpy.float64)
    bob.ip.flip(A3_org, B)
    self.assertTrue( (B == A3_ans_flip).all())
    C = bob.ip.flip(A3_org)
    self.assertTrue( (C == A3_ans_flip).all())

  def test04_flop_3D(self):
    B = numpy.ndarray((3,2,2), numpy.float64)
    bob.ip.flop(A3_org, B)
    self.assertTrue( (B == A3_ans_flop).all())
    C = bob.ip.flop(A3_org)
    self.assertTrue( (C == A3_ans_flop).all())

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(FlipFlopTest)
