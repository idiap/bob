#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import cv
import os, sys
import unittest
import torch

class FilterNewTest(unittest.TestCase):
  """Performs various combined filter tests."""
  def test01_rotate(self):
    print ""

    img = torch.database.Array('data/faceextract/test_001.png')
    A = img.get()[1,:,:] ## get the gray plane as blitzarray

    ## 
    # B = A.sameAs();
    
if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()

