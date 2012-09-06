#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Apr 20 17:32:54 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Basic tests for the error measuring system of bob
"""

import os, sys
import unittest
import numpy
import bob
import pkg_resources

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def count(array, value=True):
  """Counts occurrences of a certain value in an array"""
  return list(array == value).count(True)

def save(fname, data):
  """Saves a single array into a file in the 'data' directory."""
  bob.io.Array(data).save(os.path.join('data', fname))

class ErrorTest(unittest.TestCase):
  """Various measure package tests for error evaluation."""

  def test01_basicRatios(self):

    # We test the basic functionaly on FAR and FRR calculation. The first
    # example is separable, with a separation threshold of about 3.0

    positives = bob.io.load(F('linsep-positives.hdf5'))
    negatives = bob.io.load(F('linsep-negatives.hdf5'))

    minimum = min(positives.min(), negatives.min())
    maximum = max(positives.max(), negatives.max())

    # If we take a threshold on the minimum, the FAR should be 1.0 and the FRR
    # should be 0.0.
    far, frr = bob.measure.farfrr(negatives, positives, minimum-0.1)
    self.assertEqual(far, 1.0)
    self.assertEqual(frr, 0.0)

    # Similarly, if we take a threshold on the maximum, the FRR should be 1.0
    # while the FAR should be 0.0
    far, frr = bob.measure.farfrr(negatives, positives, maximum+0.1)
    self.assertEqual(far, 0.0)
    self.assertEqual(frr, 1.0)

    # If we choose the appropriate threshold, we should get 0.0 for both FAR
    # and FRR.
    far, frr = bob.measure.farfrr(negatives, positives, 3.0)
    self.assertEqual(far, 0.0)
    self.assertEqual(frr, 0.0)

  def test02_indexing(self):

    # This test verifies that the output of correctly_classified_positives() and
    # correctly_classified_negatives() makes sense.
    positives = bob.io.load(F('linsep-positives.hdf5'))
    negatives = bob.io.load(F('linsep-negatives.hdf5'))

    minimum = min(positives.min(), negatives.min())
    maximum = max(positives.max(), negatives.max())

    # If the threshold is minimum, we should have all positive samples
    # correctly classified and none of the negative samples correctly
    # classified.
    self.assertTrue(bob.measure.correctly_classified_positives(positives,
      minimum-0.1).all())
    self.assertFalse(bob.measure.correctly_classified_negatives(negatives,
      minimum-0.1).any())

    # The inverse is true if the threshold is a bit above the maximum.
    self.assertFalse(bob.measure.correctly_classified_positives(positives,
      maximum+0.1).any())
    self.assertTrue(bob.measure.correctly_classified_negatives(negatives,
      maximum+0.1).all())

    # If the threshold separates the sets, than all should be correctly
    # classified.
    self.assertTrue(bob.measure.correctly_classified_positives(positives, 3).all())
    self.assertTrue(bob.measure.correctly_classified_negatives(negatives, 3).all())

  def test03_thresholding(self):

    # This example will demonstrate and check the use of eer_threshold() to
    # calculate the threshold that minimizes the EER.
   
    # This test set is not separable.
    positives = bob.io.load(F('nonsep-positives.hdf5'))
    negatives = bob.io.load(F('nonsep-negatives.hdf5'))
    threshold = bob.measure.eer_threshold(negatives, positives)

    # Of course we have to make sure that will set the EER correctly:
    ccp = count(bob.measure.correctly_classified_positives(positives,threshold))
    ccn = count(bob.measure.correctly_classified_negatives(negatives,threshold))
    self.assertTrue( (ccp - ccn) <= 1 )

    # If the set is separable, the calculation of the threshold is a little bit
    # trickier, as you have no points in the middle of the range to compare
    # things to. This is where the currently used recursive algorithm seems to
    # do better. Let's verify
    positives = bob.io.load(F('linsep-positives.hdf5'))
    negatives = bob.io.load(F('linsep-negatives.hdf5'))
    threshold = bob.measure.eer_threshold(negatives, positives)
    # the result here is 3.242 (which is what is expect ;-)

    # Of course we have to make sure that will set the EER correctly:
    ccp = count(bob.measure.correctly_classified_positives(positives,threshold))
    ccn = count(bob.measure.correctly_classified_negatives(negatives,threshold))
    self.assertEqual(ccp, ccn)

    # The second option for the calculation of the threshold is to use the
    # minimum HTER.
    threshold2 = bob.measure.min_hter_threshold(negatives, positives)
    # the result here is 3.242 (which is what is expect ;-)
    self.assertEqual(threshold, threshold2) #in this particular case

    # Of course we have to make sure that will set the EER correctly:
    ccp = count(bob.measure.correctly_classified_positives(positives,threshold2))
    ccn = count(bob.measure.correctly_classified_negatives(negatives,threshold2))
    self.assertEqual(ccp, ccn)

  def test04_plots(self):

    # This test set is not separable.
    positives = bob.io.load(F('nonsep-positives.hdf5'))
    negatives = bob.io.load(F('nonsep-negatives.hdf5'))
    threshold = bob.measure.eer_threshold(negatives, positives)

    # This example will test the ROC plot calculation functionality.
    xy = bob.measure.roc(negatives, positives, 100)
    # uncomment the next line to save a reference value
    # save('nonsep-roc.hdf5', xy)
    xyref = bob.io.load(F('nonsep-roc.hdf5'))
    self.assertTrue( numpy.array_equal(xy, xyref) )

    # This example will test the DET plot calculation functionality.
    det_xyzw = bob.measure.det(negatives, positives, 100)
    # uncomment the next line to save a reference value
    # save('nonsep-det.hdf5', det_xyzw)
    det_xyzw_ref = bob.io.load(F('nonsep-det.hdf5'))
    self.assertTrue( numpy.allclose(det_xyzw, det_xyzw_ref, atol=1e-15) )

    # This example will test the EPC plot calculation functionality. For the
    # EPC curve, you need to have a development and a test set. We will split,
    # by the middle, the negatives and positives sample we have, just for the
    # sake of testing
    dev_negatives = negatives[:(negatives.shape[0]/2)]
    test_negatives = negatives[(negatives.shape[0]/2):]
    dev_positives = positives[:(positives.shape[0]/2)]
    test_positives = positives[(positives.shape[0]/2):]
    xy = bob.measure.epc(dev_negatives, dev_positives, 
        test_negatives, test_positives, 100)
    # uncomment the next line to save a reference value
    # save('nonsep-epc.hdf5', xy)
    xyref = bob.io.load(F('nonsep-epc.hdf5'))
    self.assertTrue( numpy.allclose(xy, xyref, atol=1e-15) )
