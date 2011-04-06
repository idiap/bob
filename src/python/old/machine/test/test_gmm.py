#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 16 Jul 2010 09:30:53 CEST

"""Tests cropping features
"""

import os, sys
import unittest
import torch

def compare(v1, v2, width):
  return abs(v1-v2) <= width

def test_file(name):
  """Returns the path to the filename for this test."""
  return os.path.join("data", "gmm", name)

class GmmTest(unittest.TestCase):
  """Performs various tests for the Torch::ipGeomNorm object."""

  ###############################################################
  #

  def test_load_and_construct_01(self):
    gmm = torch.machine.MultiVariateDiagonalGaussianDistribution()
    status = gmm.loadFile("does.not.exist.gmm")
    self.assertFalse(status)

  def test_load_and_construct_02(self):
    gmm = torch.machine.MultiVariateDiagonalGaussianDistribution()
    status = gmm.loadFile(test_file("1001.gmm"))
    self.assertTrue(status)

  def test_load_and_construct_03(self):
    gmm = torch.machine.MultiVariateDiagonalGaussianDistribution(test_file("1001.gmm"))
    self.assertTrue(True) # bad need to find a good test here

  def test_load_and_construct_04(self):

    try:
      gmm = torch.machine.MultiVariateDiagonalGaussianDistribution("does.not.exist.gmm")
    except:
      self.assertTrue(True) # an exception was raised - good
      return                # leave test
 
    self.assertTrue(False)  # no exception was raised - wrong

  ###############################################################
  #

  def test_score_01(self):
    zero_tensor = torch.core.DoubleTensor(91)
    zero_tensor.fill(0)
    gmm = torch.machine.MultiVariateDiagonalGaussianDistribution(test_file("1001.gmm"))

    score = gmm.score(zero_tensor)
    self.assertTrue(compare(score, -15.3702381398, 1e-6))
 
  # same score but different load
  def test_score_02(self):
    zero_tensor = torch.core.DoubleTensor(91)
    zero_tensor.fill(0)

    gmm = torch.machine.MultiVariateDiagonalGaussianDistribution()
    gmm.loadFile(test_file("1001.gmm"))

    score = gmm.score(zero_tensor)
    self.assertTrue(compare(score, -15.3702381398, 1e-6))

  def test_score_03(self):
    zero_tensor = torch.core.DoubleTensor(91)
    zero_tensor.fill(0)

    gmm = torch.machine.MultiVariateDiagonalGaussianDistribution(test_file("1046.gmm"))

    score = gmm.score(zero_tensor)
    self.assertTrue(compare(score, -14.6556824302, 1e-6))

  def test_score_04(self):
    lds = torch.core.ListDataSet()
    status = lds.load(test_file("1001_f_g1_s01_1001_en_4.chris.dct"))

    # make sure we read everything
    self.assertEqual(status, 2337)

    gmm = torch.machine.MultiVariateDiagonalGaussianDistribution(test_file("1001.gmm"))
    score = gmm.score(lds)
    
    # make sure that the score is correct
    self.assertTrue(compare(score, -98.8320520107, 1e-6))


if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
