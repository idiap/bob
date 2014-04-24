#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Mon Jan 23 20:46:07 2012 +0100
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests our Dense SIFT features extractor based on VLFeat
"""

import os, sys
import unittest
import bob
import numpy
import pkg_resources
from nose.plugins.skip import SkipTest
import functools

def vldsift_found(test):
  '''Decorator to check if the VLDSIFT class is present before enabling a test'''

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      from .._ip import VLDSIFT
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest('VLFeat was not available at compile time')

  return wrapper


def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def load_image(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join("sift", relative_filename)
  array = bob.io.load(F(filename))
  return array.astype('float32')

def equal(x, y, epsilon):
  return (abs(x - y) < epsilon)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class VLDSiftTest(unittest.TestCase):
  """Performs various tests"""

  @vldsift_found
  def test01_VLDSiftPython(self):
    # Dense SIFT reference using VLFeat 0.9.13 
    # (First 3 descriptors, Gaussian window)
    filename_beg = F(os.path.join("sift", "vldsift_gref_beg.hdf5"))
    ref_vl_beg = bob.io.load(filename_beg)
    filename_end = F(os.path.join("sift", "vldsift_gref_end.hdf5"))
    ref_vl_end = bob.io.load(filename_end)

    # Computes dense SIFT feature using VLFeat binding
    img = load_image('vlimg_ref.pgm')
    mydsift1 = bob.ip.VLDSIFT(img.shape[0],img.shape[1])
    out_vl = mydsift1(img)
    # Compare to reference (first 200 descriptors)
    offset = out_vl.shape[0]-200
    for i in range(200):
      self.assertTrue(equals(out_vl[i,:], ref_vl_beg[i,:], 2e-6))
      self.assertTrue(equals(out_vl[offset+i,:], ref_vl_end[i,:], 2e-6))
