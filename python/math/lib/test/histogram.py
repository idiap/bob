#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Tue May  1 18:12:43 CEST 2012
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

"""Tests bob interior point Linear Programming solvers
"""

import os, sys
import unittest
import bob
import numpy



class HistogramMeasureTest(unittest.TestCase):

  def chi_square(self, h1, h2):
    """Computes the chi-square distance between two histograms (or histogram sequences)"""
    d = 0
    for i in range(h1.shape[0]):
      if h1[i] != h2[i]: d += (h1[i] - h2[i])**2 / (h1[i] + h2[i])
    return d
  
  
  def histogram_intersection(self, h1, h2):
    """Computes the intersection measure of the given histograms (or histogram sequences)"""
    dist = 0
    for i in range(h1.shape[0]):
      dist += min(h1[i], h2[i])
    return dist


  # initialize histograms to test the two measures  
  m_h1 = numpy.array([0,15,3,7,4,0,3,0,17,12], dtype = numpy.int32) 
  m_h2 = numpy.array([2,7,14,3,25,0,7,1,0,4], dtype = numpy.int32)
  
  m_h3 = numpy.random.random_integers(0,99,size=(100000,))
  m_h4 = numpy.random.random_integers(0,99,size=(100000,))
  
  m_h5 = numpy.array([1,0,0,1,0,0,1,0,1,1], dtype = numpy.float64);
  m_h6 = numpy.array([1,0,1,0,0,0,1,0,1,1], dtype = numpy.float64);

  
  
  def test_histogram_intersection(self):
    # compare our implementation with bob.math
    self.assertEqual(
      self.histogram_intersection(self.m_h1, self.m_h2),
      bob.math.histogram_intersection(self.m_h1, self.m_h2)
    )
    self.assertEqual(
      self.histogram_intersection(self.m_h3, self.m_h4),
      bob.math.histogram_intersection(self.m_h3, self.m_h4)
    )
    
    # test specific (simple) case
    self.assertEqual(bob.math.histogram_intersection(self.m_h5, self.m_h5), 5.)
    self.assertEqual(bob.math.histogram_intersection(self.m_h5, self.m_h6), 4.)
    
    
  def test_chi_square(self):
    # compare our implementation with bob.math
    self.assertEqual(
      self.chi_square(self.m_h1, self.m_h2),
      bob.math.chi_square(self.m_h1, self.m_h2)
    )    
    self.assertEqual(
      self.chi_square(self.m_h3, self.m_h4),
      bob.math.chi_square(self.m_h3, self.m_h4)
    )    

    # test specific (simple) case
    self.assertEqual(bob.math.chi_square(self.m_h5, self.m_h5), 0.)
    self.assertEqual(bob.math.chi_square(self.m_h5, self.m_h6), 2.)
    
# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(HistogramMeasureTest)
