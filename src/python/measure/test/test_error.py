#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 20 Apr 2011 15:30:23 CEST 

"""Basic tests for the error measuring system of bob
"""

import os, sys
import unittest
import numpy
import bob

def load(fname):
  """Loads a single array from the 'data' directory."""
  return bob.io.Array(os.path.join('data', fname)).get()

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

    positives = load('linsep-positives.hdf5')
    negatives = load('linsep-negatives.hdf5')

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

    # This test verifies that the output of correctlyClassifiedPositives() and
    # correctlyClassifiedPositives() makes sense.
    positives = load('linsep-positives.hdf5')
    negatives = load('linsep-negatives.hdf5')

    minimum = min(positives.min(), negatives.min())
    maximum = max(positives.max(), negatives.max())

    # If the threshold is minimum, we should have all positive samples
    # correctly classified and none of the negative samples correctly
    # classified.
    self.assertTrue(bob.measure.correctlyClassifiedPositives(positives,
      minimum-0.1).all())
    self.assertFalse(bob.measure.correctlyClassifiedNegatives(negatives,
      minimum-0.1).any())

    # The inverse is true if the threshold is a bit above the maximum.
    self.assertFalse(bob.measure.correctlyClassifiedPositives(positives,
      maximum+0.1).any())
    self.assertTrue(bob.measure.correctlyClassifiedNegatives(negatives,
      maximum+0.1).all())

    # If the threshold separates the sets, than all should be correctly
    # classified.
    self.assertTrue(bob.measure.correctlyClassifiedPositives(positives, 3).all())
    self.assertTrue(bob.measure.correctlyClassifiedNegatives(negatives, 3).all())

  def test03_thresholding(self):

    # This example will demonstrate and check the use of eerThreshold() to
    # calculate the threshold that minimizes the EER.
   
    # This test set is not separable.
    positives = load('nonsep-positives.hdf5')
    negatives = load('nonsep-negatives.hdf5')
    threshold = bob.measure.eerThreshold(negatives, positives)

    # Of course we have to make sure that will set the EER correctly:
    ccp = count(bob.measure.correctlyClassifiedPositives(positives,threshold))
    ccn = count(bob.measure.correctlyClassifiedNegatives(negatives,threshold))
    self.assertTrue( (ccp - ccn) <= 1 )

    # If the set is separable, the calculation of the threshold is a little bit
    # trickier, as you have no points in the middle of the range to compare
    # things to. This is where the currently used recursive algorithm seems to
    # do better. Let's verify
    positives = load('linsep-positives.hdf5')
    negatives = load('linsep-negatives.hdf5')
    threshold = bob.measure.eerThreshold(negatives, positives)
    # the result here is 3.242 (which is what is expect ;-)

    # Of course we have to make sure that will set the EER correctly:
    ccp = count(bob.measure.correctlyClassifiedPositives(positives,threshold))
    ccn = count(bob.measure.correctlyClassifiedNegatives(negatives,threshold))
    self.assertEqual(ccp, ccn)

    # The second option for the calculation of the threshold is to use the
    # minimum HTER.
    threshold2 = bob.measure.minHterThreshold(negatives, positives)
    # the result here is 3.242 (which is what is expect ;-)
    self.assertEqual(threshold, threshold2) #in this particular case

    # Of course we have to make sure that will set the EER correctly:
    ccp = count(bob.measure.correctlyClassifiedPositives(positives,threshold2))
    ccn = count(bob.measure.correctlyClassifiedNegatives(negatives,threshold2))
    self.assertEqual(ccp, ccn)

  def test04_plots(self):

    # This test set is not separable.
    positives = load('nonsep-positives.hdf5')
    negatives = load('nonsep-negatives.hdf5')
    threshold = bob.measure.eerThreshold(negatives, positives)

    # This example will test the ROC plot calculation functionality.
    xy = bob.measure.roc(negatives, positives, 100)
    # uncomment the next line to save a reference value
    # save('nonsep-roc.hdf5', xy)
    xyref = load('nonsep-roc.hdf5')
    self.assertTrue( numpy.array_equal(xy, xyref) )

    # This example will test the DET plot calculation functionality.
    det_xyzw = bob.measure.det(negatives, positives, 100)
    # uncomment the next line to save a reference value
    # save('nonsep-det.hdf5', det_xyzw)
    det_xyzw_ref = load('nonsep-det.hdf5')
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
    xyref = load('nonsep-epc.hdf5')
    self.assertTrue( numpy.allclose(xy, xyref, atol=1e-15) )

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()
