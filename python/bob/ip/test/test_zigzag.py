#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Niklas Johansson <niklas.johansson@idiap.ch>
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Apr 6 14:16:13 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

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
