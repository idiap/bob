#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 16 Jul 2010 09:30:53 CEST

"""Tests cropping features
"""

import os, sys
import unittest
import torch

def test_file(name):
  """Returns the path to the filename for this test."""
  return os.path.join("data", name)

class ListDataSetTest(unittest.TestCase):
  """Performs a few tests tests for the ListDataSet type."""

  def test_load(self):
    lds = torch.core.ListDataSet()
    status = lds.load(test_file("1001_f_g1_s01_1001_en_4.chris.dct"))

    # make sure we read everything
    self.assertEqual(status, 2337)

if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()

