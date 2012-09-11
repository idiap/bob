#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
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

"""Tests the reference counting of the C++-Python array  bridge.
"""

import unittest
import bob 
import numpy
import sys

class NumpyRefcountTest(unittest.TestCase):
  """Performs various conversion tests."""
  
  def test01_refcount(self):

    for i in range(10):
      frame = numpy.random.randint(0,255,size=1000)
      self.assertTrue( sys.getrefcount(frame) == 2)
      x = bob.core.convert(frame,numpy.float64) #just use frame
      self.assertTrue( sys.getrefcount(frame) == 2)

