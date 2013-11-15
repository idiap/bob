#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Nov 18 14:16:13 2011 +0100
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests some functionality of the C++-Python array conversion bridge.
"""

import unittest
import bob
import numpy

class ConversionTest(unittest.TestCase):
  """Performs various conversion tests."""
  
  def xtest01_default_ranges(self):

    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.convert(x, 'uint16')
    self.assertTrue( numpy.array_equal(x.astype('uint16'), c) )

  def xtest02_from_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.convert(x, 'uint16', source_range=(0,255))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

  def test03_to_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.convert(x, 'float64', dest_range=(0.,255.))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

  def test04_from_and_to_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = bob.core.convert(x, 'float64', source_range=(0,255),
        dest_range=(0.,255.))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )
