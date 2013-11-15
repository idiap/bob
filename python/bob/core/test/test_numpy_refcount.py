#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

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

