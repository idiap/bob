#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Niklas Johansson <niklas.johansson@idiap.ch>
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Apr 6 14:16:13 2011 +0200
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

"""Test the zigzag extractor
"""

import os, sys
import unittest
import bob
import numpy

A_org    = numpy.array(range(1,17), 'float64').reshape((4,4))
A_ans_3  = numpy.array((1, 2, 5), 'float64')
A_ans_6  = numpy.array((1, 2, 5, 9, 6, 3), 'float64')
A_ans_10 = numpy.array((1, 2, 5, 9, 6, 3, 4, 7, 10, 13), 'float64')

class ZigzagTest(unittest.TestCase):
  """Performs various tests for the zigzag extractor."""
  def test01_zigzag(self):
    B = numpy.array((0, 0, 0), 'float64')
    bob.ip.zigzag(A_org, B)
    self.assertTrue( (B == A_ans_3).all())
    C = bob.ip.zigzag(A_org, 3)
    self.assertTrue( (C == A_ans_3).all())
    
  def test02_zigzag(self):
    B = numpy.array((0, 0, 0, 0, 0, 0), 'float64')
    bob.ip.zigzag(A_org, B)
    self.assertTrue( (B == A_ans_6).all())
    C = bob.ip.zigzag(A_org, 6)
    self.assertTrue( (C == A_ans_6).all())

  def test03_zigzag(self):
    B = numpy.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'float64')
    bob.ip.zigzag(A_org, B)
    self.assertTrue( (B == A_ans_10).all())
    C = bob.ip.zigzag(A_org, 10)
    self.assertTrue( (C == A_ans_10).all())
