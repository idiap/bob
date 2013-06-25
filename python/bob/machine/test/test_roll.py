#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Jun 25 20:44:40 CEST 2013
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

"""Tests on the roll/unroll functions
"""

import unittest
import bob
import numpy

class RollTest(unittest.TestCase):
  """Performs various roll/unroll tests."""

  def test01_roll(self):
    m = bob.machine.MLP((10,3,8,5))
    m.randomize()
    vec = bob.machine.unroll(m)
    m2 = bob.machine.MLP((10,3,8,5))
    bob.machine.roll(m2, vec)

    self.assertTrue( m == m2 )

  def test02_roll(self):
    w = [numpy.array([[2,3.]]), numpy.array([[2,3,4.],[5,6,7]])]
    b = [numpy.array([5.,]), numpy.array([7,8.])]
    vec = numpy.ndarray(11, numpy.float64)
    bob.machine.unroll(w, b, vec)

    w_ = [numpy.ndarray((1,2), numpy.float64), numpy.ndarray((2,3), numpy.float64)]
    b_ = [numpy.ndarray(1, numpy.float64), numpy.ndarray(2, numpy.float64)]
    bob.machine.roll(w_, b_, vec)

    self.assertTrue( (w_[0] == w[0]).all() )
    self.assertTrue( (b_[0] == b[0]).all() )
  
