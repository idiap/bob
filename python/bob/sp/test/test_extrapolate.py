#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Sep 27 23:26:46 2011 +0200
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

import os, sys
import unittest
import numpy
from .. import *

#############################################################################
# Tests blitz-based extrapolation implementation with values returned 
#############################################################################

########################## Values used for the computation ##################
eps = 1e-3
a5 = numpy.array([1,2,3,4,5], 'float64')
a14_zeros = numpy.array([0,0,0,0,1,2,3,4,5,0,0,0,0,0], 'float64')
a14_twos = numpy.array([2,2,2,2,1,2,3,4,5,2,2,2,2,2], 'float64')
a14_nearest = numpy.array([1,1,1,1,1,2,3,4,5,5,5,5,5,5], 'float64')
a14_circular = numpy.array([2,3,4,5,1,2,3,4,5,1,2,3,4,5], 'float64')
a14_mirror = numpy.array([4,3,2,1,1,2,3,4,5,5,4,3,2,1], 'float64')

a26_zeros = numpy.array([0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,0,0,0,0,0,0,0,0,0,0,0], 'float64')
a26_twos = numpy.array([2,2,2,2,2,2,2,2,2,2,1,2,3,4,5,2,2,2,2,2,2,2,2,2,2,2], 'float64')
a26_nearest = numpy.array([1,1,1,1,1,1,1,1,1,1,1,2,3,4,5,5,5,5,5,5,5,5,5,5,5,5], 'float64')
a26_circular = numpy.array([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1], 'float64')
a26_mirror = numpy.array([1,2,3,4,5,5,4,3,2,1,1,2,3,4,5,5,4,3,2,1,1,2,3,4,5,5], 'float64')

A22 = numpy.array([1,2,3,4], 'float64').reshape(2,2)
A44_zeros = numpy.array([0,0,0,0,0,1,2,0,0,3,4,0,0,0,0,0], 'float64').reshape(4,4)
A44_twos = numpy.array([2,2,2,2,2,1,2,2,2,3,4,2,2,2,2,2], 'float64').reshape(4,4)
A44_nearest = numpy.array([1,1,2,2,1,1,2,2,3,3,4,4,3,3,4,4], 'float64').reshape(4,4)
A44_circular = numpy.array([4,3,4,3,2,1,2,1,4,3,4,3,2,1,2,1], 'float64').reshape(4,4)
A44_mirror = numpy.array([1,1,2,2,1,1,2,2,3,3,4,4,3,3,4,4], 'float64').reshape(4,4)

A1111_zeros = numpy.array([0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,1,2,0,0,0,0,0,
                           0,0,0,0,3,4,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0], 'float64').reshape(11,11)
A1111_twos = numpy.array([2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,1,2,2,2,2,2,2,
                          2,2,2,2,3,4,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2,
                          2,2,2,2,2,2,2,2,2,2,2], 'float64').reshape(11,11)
A1111_nearest = numpy.array([1,1,1,1,1,2,2,2,2,2,2,
                             1,1,1,1,1,2,2,2,2,2,2,
                             1,1,1,1,1,2,2,2,2,2,2,
                             1,1,1,1,1,2,2,2,2,2,2,
                             1,1,1,1,1,2,2,2,2,2,2,
                             3,3,3,3,3,4,4,4,4,4,4,
                             3,3,3,3,3,4,4,4,4,4,4,
                             3,3,3,3,3,4,4,4,4,4,4,
                             3,3,3,3,3,4,4,4,4,4,4,
                             3,3,3,3,3,4,4,4,4,4,4,
                             3,3,3,3,3,4,4,4,4,4,4], 'float64').reshape(11,11)
A1111_circular = numpy.array([1,2,1,2,1,2,1,2,1,2,1,
                              3,4,3,4,3,4,3,4,3,4,3,
                              1,2,1,2,1,2,1,2,1,2,1,
                              3,4,3,4,3,4,3,4,3,4,3,
                              1,2,1,2,1,2,1,2,1,2,1,
                              3,4,3,4,3,4,3,4,3,4,3,
                              1,2,1,2,1,2,1,2,1,2,1,
                              3,4,3,4,3,4,3,4,3,4,3,
                              1,2,1,2,1,2,1,2,1,2,1,
                              3,4,3,4,3,4,3,4,3,4,3,
                              1,2,1,2,1,2,1,2,1,2,1], 'float64').reshape(11,11)
A1111_mirror = numpy.array([1,2,2,1,1,2,2,1,1,2,2,
                            3,4,4,3,3,4,4,3,3,4,4,
                            3,4,4,3,3,4,4,3,3,4,4,
                            1,2,2,1,1,2,2,1,1,2,2,
                            1,2,2,1,1,2,2,1,1,2,2,
                            3,4,4,3,3,4,4,3,3,4,4,
                            3,4,4,3,3,4,4,3,3,4,4,
                            1,2,2,1,1,2,2,1,1,2,2,
                            1,2,2,1,1,2,2,1,1,2,2,
                            3,4,4,3,3,4,4,3,3,4,4,
                            3,4,4,3,3,4,4,3,3,4,4], 'float64').reshape(11,11)

#############################################################################


def compare(v1, v2, width):
  return abs(v1-v2) <= width

def _extrapolate_1D(res, reference, obj):
  # Tests the extrapolation
  obj.assertEqual(res.shape, reference.shape)
  for i in range(res.shape[0]):
    obj.assertTrue(compare(res[i], reference[i], eps))

def _extrapolate_2D(res, reference, obj):
  # Tests the extrapolation
  obj.assertEqual(res.shape, reference.shape)
  for i in range(res.shape[0]):
    for j in range(res.shape[1]):
      obj.assertTrue(compare(res[i,j], reference[i,j], eps))


##################### Unit Tests ##################  
class ExtrapolationTest(unittest.TestCase):
  """Performs extrapolation product"""

##################### Convolution Tests ##################  
  def test_extrapolation_1D_zeros(self):
    b = numpy.zeros((14,), numpy.float64)
    extrapolate_zero(a5,b)
    _extrapolate_1D(b,a14_zeros,self)

    b = numpy.zeros((26,), numpy.float64)
    extrapolate_zero(a5,b)
    _extrapolate_1D(b,a26_zeros,self)

  def test_extrapolation_1D_twos(self):
    b = numpy.zeros((14,), numpy.float64)
    extrapolate_constant(a5,b,2.)
    _extrapolate_1D(b,a14_twos,self)

    b = numpy.zeros((26,), numpy.float64)
    extrapolate_constant(a5,b,2.)
    _extrapolate_1D(b,a26_twos,self)

  def test_extrapolation_1D_nearest(self):
    b = numpy.zeros((14,), numpy.float64)
    extrapolate_nearest(a5,b)
    _extrapolate_1D(b,a14_nearest,self)

    b = numpy.zeros((26,), numpy.float64)
    extrapolate_nearest(a5,b)
    _extrapolate_1D(b,a26_nearest,self)

  def test_extrapolation_1D_circular(self):
    b = numpy.zeros((14,), numpy.float64)
    extrapolate_circular(a5,b)
    _extrapolate_1D(b,a14_circular,self)

    b = numpy.zeros((26,), numpy.float64)
    extrapolate_circular(a5,b)
    _extrapolate_1D(b,a26_circular,self)

  def test_extrapolation_1D_mirror(self):
    b = numpy.zeros((14,), numpy.float64)
    extrapolate_mirror(a5,b)
    _extrapolate_1D(b,a14_mirror,self)

    b = numpy.zeros((26,), numpy.float64)
    extrapolate_mirror(a5,b)
    _extrapolate_1D(b,a26_mirror,self)

  def test_extrapolation_2D_zeros(self):
    B = numpy.zeros((4,4), numpy.float64)
    extrapolate_zero(A22,B)
    _extrapolate_2D(B,A44_zeros,self)

    B = numpy.zeros((11,11), numpy.float64)
    extrapolate_zero(A22,B)
    _extrapolate_2D(B,A1111_zeros,self)

  def test_extrapolation_2D_twos(self):
    B = numpy.zeros((4,4), numpy.float64)
    extrapolate_constant(A22,B,2.)
    _extrapolate_2D(B,A44_twos,self)

    B = numpy.zeros((11,11), numpy.float64)
    extrapolate_constant(A22,B,2.)
    _extrapolate_2D(B,A1111_twos,self)

  def test_extrapolation_2D_nearest(self):
    B = numpy.zeros((4,4), numpy.float64)
    extrapolate_nearest(A22,B)
    _extrapolate_2D(B,A44_nearest,self)
  
    B = numpy.zeros((11,11), numpy.float64)
    extrapolate_nearest(A22,B)
    _extrapolate_2D(B,A1111_nearest,self)
  
  def test_extrapolation_2D_circular(self):
    B = numpy.zeros((4,4), numpy.float64)
    extrapolate_circular(A22,B)
    _extrapolate_2D(B,A44_circular,self)

    B = numpy.zeros((11,11), numpy.float64)
    extrapolate_circular(A22,B)
    _extrapolate_2D(B,A1111_circular,self)

  def test_extrapolation_2D_mirror(self):
    B = numpy.zeros((4,4), numpy.float64)
    extrapolate_mirror(A22,B)
    _extrapolate_2D(B,A44_mirror,self)

    B = numpy.zeros((11,11), numpy.float64)
    extrapolate_mirror(A22,B)
    _extrapolate_2D(B,A1111_mirror,self)
