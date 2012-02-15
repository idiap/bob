#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Nov 18 14:16:13 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

"""Tests some functionality of the C++-Python array conversion bridge.
"""

import os
import sys
import unittest
import bob
import numpy

class ConversionTest(unittest.TestCase):
  """Performs various conversion tests."""
  
  def xtest01_default_ranges(self):

    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.array.convert(x, 'uint16')
    self.assertTrue( numpy.array_equal(x.astype('uint16'), c) )

  def xtest02_from_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.array.convert(x, 'uint16', sourceRange=(0,255))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

  def test03_to_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.array.convert(x, 'float64', destRange=(0.,255.))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

  def test04_from_and_to_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.array.convert(x, 'float64', sourceRange=(0,255),
        destRange=(0.,255.))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

def main():
  unittest.main()
