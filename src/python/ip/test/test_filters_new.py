#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import os, sys
import unittest
import bob
import numpy

A_org    = numpy.array(range(1,17), 'float64').reshape((4,4))
A_ans_3  = numpy.array((1, 2, 5), 'float64')
A_ans_6  = numpy.array((1, 2, 5, 9, 6, 3), 'float64')
A_ans_10 = numpy.array((1, 2, 5, 9, 6, 3, 4, 7, 10, 13), 'float64')

class FilterNewTest(unittest.TestCase):
  """Performs various combined filter tests."""
  def test01_zigzag(self):

    B = numpy.array((0, 0, 0), 'float64')
    bob.ip.zigzag(A_org, B, 3)

    self.assertEqual( (B == A_ans_3).all(), True)
    
  def test02_zigzag(self):

    B = numpy.array((0, 0, 0, 0, 0, 0), 'float64')
    bob.ip.zigzag(A_org, B, 6)

    self.assertEqual( (B == A_ans_6).all(), True)

  def test03_zigzag(self):

    B = numpy.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'float64')
    bob.ip.zigzag(A_org, B, 10)

    self.assertEqual( (B == A_ans_10).all(), True)

if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()

