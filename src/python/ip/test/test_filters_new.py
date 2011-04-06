#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import os, sys
import unittest
import torch

A_org    = torch.core.array.float64_2((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), (4,4))
A_ans_3  = torch.core.array.float64_1((1, 2, 5), (3,))
A_ans_6  = torch.core.array.float64_1((1, 2, 5, 9, 6, 3), (6,))
A_ans_10 = torch.core.array.float64_1((1, 2, 5, 9, 6, 3, 4, 7, 10, 13), (10,))

class FilterNewTest(unittest.TestCase):
  """Performs various combined filter tests."""
  def test01_zigzag(self):

    B = torch.core.array.float64_1((0, 0, 0), (3,))
    torch.ip.zigzag(A_org, B, 3)

    self.assertEqual( (B == A_ans_3).all(), True)
    
  def test02_zigzag(self):

    B = torch.core.array.float64_1((0, 0, 0, 0, 0, 0), (6,))
    torch.ip.zigzag(A_org, B, 6)

    self.assertEqual( (B == A_ans_6).all(), True)

  def test03_zigzag(self):

    B = torch.core.array.float64_1((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (10,))
    torch.ip.zigzag(A_org, B, 10)

    self.assertEqual( (B == A_ans_10).all(), True)

if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()

