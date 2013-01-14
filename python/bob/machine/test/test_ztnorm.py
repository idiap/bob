#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Tue Jul 19 15:33:20 2011 +0200
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

"""Tests on the ZTNorm function
"""

import os, sys
import unittest
import numpy
import bob
import pkg_resources

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def sameValue(vect_A, vect_B):
  sameMatrix = numpy.zeros((vect_A.shape[0], vect_B.shape[0]), 'bool')

  for j in range(vect_A.shape[0]):
    for i in range(vect_B.shape[0]):
      sameMatrix[j, i] = (vect_A[j] == vect_B[i])

  return sameMatrix
  
class ZTNormTest(unittest.TestCase):
  """Performs various ZTNorm tests."""

  def test01_ztnorm_simple(self):
    # 3x5
    my_A = numpy.array([[1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 8],
                        [7, 6, 5, 4, 3]],'float64')
    # 3x4
    my_B = numpy.array([[5, 4, 7, 8],[9, 8, 7, 4],[5, 6, 3, 2]],'float64')
    # 2x5
    my_C = numpy.array([[5, 4, 3, 2, 1],[2, 1, 2, 3, 4]],'float64')
    # 2x4
    my_D = numpy.array([[8, 6, 4, 2],[0, 2, 4, 6]],'float64')
    
    # 4x1
    znorm_id = numpy.array([1, 2, 3, 4],'uint32')
    # 2x1
    tnorm_id = numpy.array([1, 5],'uint32')

    scores = bob.machine.ztnorm(my_A, my_B, my_C, my_D,
        sameValue(tnorm_id, znorm_id))

    ref_scores = numpy.array([[-4.45473107e+00, -3.29289322e+00, -1.50519101e+01, -8.42086557e-01, 6.46544511e-03], [-8.27619927e-01,  7.07106781e-01,  1.13757710e+01,  2.01641412e+00, 7.63765080e-01], [ 2.52913570e+00,  2.70710678e+00,  1.24400233e+01,  7.07106781e-01, 6.46544511e-03]], 'float64')
    
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

  def test02_ztnorm_big(self):
    my_A = bob.io.load(F("ztnorm_eval_eval.mat"))
    my_B = bob.io.load(F("ztnorm_znorm_eval.mat"))
    my_C = bob.io.load(F("ztnorm_eval_tnorm.mat"))
    my_D = bob.io.load(F("ztnorm_znorm_tnorm.mat"))

    ref_scores = bob.io.load(F("ztnorm_result.mat"))
    scores = bob.machine.ztnorm(my_A, my_B, my_C, my_D)
    
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())
