#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import os, sys
import unittest
import torch

class FilterNewTest(unittest.TestCase):
  """Performs various combined filter tests."""

  def test01_shiftColorImage(self):
    print ""

    img = torch.database.Array(os.path.join('data', 'faceextract', 'test_001.png'))
    A = img.get()
    B = A.copy()

    delta_h = 100
    delta_w = 100

    # shift to center
    torch.ip.shift(A, B, delta_h, delta_w);

    # save image
    torch.database.Array(B).save(os.path.join('data', 'faceextract', 'test_001.shift.png'));

  def test02_shiftToCenterBlue(self):
    print ""

    img = torch.database.Array(os.path.join('data', 'faceextract', 'test_001.png'))
    A = img.get()
    B = A.sameAs();

    # shift to center
    torch.ip.shiftToCenterOfPoints(A, B, 120, 147, 90, 213)

    # save image
    torch.database.Array(B).save(os.path.join('data', 'faceextract', 'test_001.blue.answer.png'));

if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()

