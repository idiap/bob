#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon Jun 20 16:15:36 2011 +0200
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

"""Tests for statistical methods
"""

import os, sys
import unittest
import bob
import numpy

class StatsTest(unittest.TestCase):
  """Tests some statistical APIs for bob"""

  def setUp(self):

    self.data = numpy.vstack(bob.db.iris.data()['setosa'])
 
  def test01_scatter(self):

    # This test demonstrates how to use the scatter matrix function of bob.
    S, M = bob.math.scatter(self.data.T)
    S /= (self.data.shape[1]-1)

    # Do the same with numpy and compare. Note that with numpy we are computing
    # the covariance matrix which is the scatter matrix divided by (N-1).
    K = numpy.array(numpy.cov(self.data))
    self.assertTrue( (abs(S-K) < 1e-10).all() )
