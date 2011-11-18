#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 18 Nov 09:49:50 2011 CET

"""Tests some functionality of the C++-Python array conversion bridge.
"""

import os
import sys
import unittest
import torch
import numpy

class ConversionTest(unittest.TestCase):
  """Performs various conversion tests."""
  
  def xtest01_default_ranges(self):

    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = torch.core.array.convert(x, 'uint16')
    self.assertTrue( numpy.array_equal(x.astype('uint16'), c) )

  def xtest02_from_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = torch.core.array.convert(x, 'uint16', sourceRange=(0,255))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

  def test03_to_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = torch.core.array.convert(x, 'float64', destRange=(0.,255.))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

  def test04_from_and_to_range(self):
  
    x = numpy.array(range(6), 'uint8').reshape(2,3)
    c = torch.core.array.convert(x, 'float64', sourceRange=(0,255),
        destRange=(0.,255.))
    self.assertTrue( numpy.array_equal(x.astype('float64'), c) )

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
