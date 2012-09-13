#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Aug 24 18:48:00 2012 +0200
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

"""Tests the TanTriggs filter
"""

import os, sys
import unittest
import bob
import numpy


eps = 1e-4

class TanTriggsTest(unittest.TestCase):
  """Performs various tests for the bob.ip.TanTriggs class"""

  def test01_parametrization(self):
    # Parametrization tests
    op = bob.ip.TanTriggs(0.2,1.,2.,2,10.,0.1)
    self.assertEqual(op.gamma, 0.2)
    self.assertEqual(op.sigma0, 1.)
    self.assertEqual(op.sigma1, 2.)
    self.assertEqual(op.radius, 2)
    self.assertEqual(op.threshold, 10.)
    self.assertEqual(op.alpha, 0.1)
    self.assertEqual(op.conv_border, bob.sp.BorderType.Mirror)
    op.gamma = 0.1
    op.sigma0 = 2.
    op.sigma1 = 3.
    op.radius = 3
    op.threshold = 8.
    op.alpha = 0.2
    op.conv_border = bob.sp._sp.BorderType.Circular
    self.assertEqual(op.gamma, 0.1)
    self.assertEqual(op.sigma0, 2.)
    self.assertEqual(op.sigma1, 3.)
    self.assertEqual(op.radius, 3)
    self.assertEqual(op.threshold, 8.)
    self.assertEqual(op.alpha, 0.2)
    self.assertEqual(op.conv_border, bob.sp.BorderType.Circular)
    op.reset(0.2,1.,2.,2,10.,0.1,bob.sp.BorderType.Mirror)
    self.assertEqual(op.gamma, 0.2)
    self.assertEqual(op.sigma0, 1.)
    self.assertEqual(op.sigma1, 2.)
    self.assertEqual(op.radius, 2)
    self.assertEqual(op.threshold, 10.)
    self.assertEqual(op.alpha, 0.1)
    self.assertEqual(op.conv_border, bob.sp.BorderType.Mirror)

  """  
  def test02_processing(self):
    # Processing tests
    # TODO (also performed in the C++ part)
  """

  def test03_comparison(self):
    # Comparisons tests
    op1 = bob.ip.TanTriggs(0.2,1.,2.,2,10.,0.1)
    op1b = bob.ip.TanTriggs(0.2,1.,2.,2,10.,0.1)
    op2 = bob.ip.TanTriggs(0.2,1.,2.,2,10.,0.1, bob.sp.BorderType.Circular)
    op3 = bob.ip.TanTriggs(0.2,1.,2.,2,10.,0.2)
    op4 = bob.ip.TanTriggs(0.2,1.,2.,2,8.,0.1)
    op5 = bob.ip.TanTriggs(0.2,1.,2.,3,10.,0.1)
    op6 = bob.ip.TanTriggs(0.2,1.,3.,2,10.,0.1)
    op7 = bob.ip.TanTriggs(0.2,1.5,2.,2,10.,0.1)
    op8 = bob.ip.TanTriggs(0.1,1.,2.,2,10.,0.1)
    self.assertEqual(op1 == op1, True)
    self.assertEqual(op1 == op1b, True)
    self.assertEqual(op1 == op2, False)
    self.assertEqual(op1 == op3, False)
    self.assertEqual(op1 == op4, False)
    self.assertEqual(op1 == op5, False)
    self.assertEqual(op1 == op6, False)
    self.assertEqual(op1 == op7, False)
    self.assertEqual(op1 == op8, False)
    self.assertEqual(op1 != op1, False)
    self.assertEqual(op1 != op1b, False)
    self.assertEqual(op1 != op2, True)
    self.assertEqual(op1 != op3, True)
    self.assertEqual(op1 != op4, True)
    self.assertEqual(op1 != op5, True)
    self.assertEqual(op1 != op6, True)
    self.assertEqual(op1 != op7, True)
    self.assertEqual(op1 != op8, True)
