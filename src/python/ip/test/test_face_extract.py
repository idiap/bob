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

def load_gray(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join("data", "flow", relative_filename)
  array = torch.database.Array(filename)
  return array.get()[0,:,:] 


class FilterNewTest(unittest.TestCase):
  """Performs various combined filter tests."""
  def test01_rotate(self):
    print ""

    
    v = torch.database.VideoReader()

if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()

