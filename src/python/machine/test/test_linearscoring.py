#!/usr/bin/env python

"""Tests on the LinearScoring function
"""

import os, sys
import unittest
import torch
import numpy

class LinearScoringTest(unittest.TestCase):
  """Performs various LinearScoring tests."""

  def test01_LinearScoring(self):
    ubm = torch.machine.GMMMachine(2, 2)
    ubm.weights   = numpy.array([0.5, 0.5], 'float64')
    ubm.means     = numpy.array([[3, 70], [4, 72]], 'float64')
    ubm.variances = numpy.array([[1, 10], [2, 5]], 'float64')
    ubm.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    model1 = torch.machine.GMMMachine(2, 2)
    model1.weights   = numpy.array([0.5, 0.5], 'float64')
    model1.means     = numpy.array([[1, 2], [3, 4]], 'float64')
    model1.variances = numpy.array([[9, 10], [11, 12]], 'float64')
    model1.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    model2 = torch.machine.GMMMachine(2, 2)
    model2.weights   = numpy.array([0.5, 0.5], 'float64')
    model2.means     = numpy.array([[5, 6], [7, 8]], 'float64')
    model2.variances = numpy.array([[13, 14], [15, 16]], 'float64')
    model2.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    stats1 = torch.machine.GMMStats(2, 2)
    stats1.sumPx = numpy.array([[1, 2], [3, 4]], 'float64')
    stats1.n = numpy.array([1, 2], 'float64')
    
    stats2 = torch.machine.GMMStats(2, 2)
    stats2.sumPx = numpy.array([[5, 6], [7, 8]], 'float64')
    stats2.n = numpy.array([3, 4], 'float64')
    
    stats3 = torch.machine.GMMStats(2, 2)
    stats3.sumPx = numpy.array([[5, 6], [7, 3]], 'float64')
    stats3.n = numpy.array([3, 4], 'float64')

    scores = torch.machine.linearScoring([model1, model2], ubm, [stats1, stats2, stats3])
    ref_scores = numpy.array([[2372.9, 5207.7, 5275.7], [2215.7, 4868.1, 4932.1]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

    scores = torch.machine.linearScoring([model1, model2], ubm, [stats1, stats2, stats3], None, True)
    ref_scores = numpy.array([[395.48333333, 371.97857143, 376.83571429],[369.28333333, 347.72142857, 352.29285714]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

    test_channeloffset = numpy.array([[9, 8, 7, 6], [5, 4, 3, 2], [1, 0, 1, 2]], 'float64')
    scores = torch.machine.linearScoring([model1, model2], ubm, [stats1, stats2, stats3], test_channeloffset, True)
    ref_scores = numpy.array([[435.91666667, 388.15, 385.17857143], [396.91666667, 357.09285714, 358.75]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())


  def test02_LinearScoring_NEW(self):
    ubm = torch.machine.GMMMachine(2, 2)
    ubm.weights   = numpy.array([0.5, 0.5], 'float64')
    ubm.means     = numpy.array([[3, 70], [4, 72]], 'float64')
    ubm.variances = numpy.array([[1, 10], [2, 5]], 'float64')
    ubm.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    model1 = torch.machine.GMMMachine(2, 2)
    model1.weights   = numpy.array([0.5, 0.5], 'float64')
    model1.means     = numpy.array([[1, 2], [3, 4]], 'float64')
    model1.variances = numpy.array([[9, 10], [11, 12]], 'float64')
    model1.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    model2 = torch.machine.GMMMachine(2, 2)
    model2.weights   = numpy.array([0.5, 0.5], 'float64')
    model2.means     = numpy.array([[5, 6], [7, 8]], 'float64')
    model2.variances = numpy.array([[13, 14], [15, 16]], 'float64')
    model2.varianceThresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    stats1 = torch.machine.GMMStats(2, 2)
    stats1.sumPx = numpy.array([[1, 2], [3, 4]], 'float64')
    stats1.n = numpy.array([1, 2], 'float64')
    stats1.T = 1+2
    
    stats2 = torch.machine.GMMStats(2, 2)
    stats2.sumPx = numpy.array([[5, 6], [7, 8]], 'float64')
    stats2.n = numpy.array([3, 4], 'float64')
    stats2.T = 3+4
    
    stats3 = torch.machine.GMMStats(2, 2)
    stats3.sumPx = numpy.array([[5, 6], [7, 3]], 'float64')
    stats3.n = numpy.array([3, 4], 'float64')
    stats3.T = 3+4

    # with channeloffset
    test_channeloffset = [numpy.array([9, 8, 7, 6], 'float64'), numpy.array([5, 4, 3, 2], 'float64'), numpy.array([1, 0, 1, 2], 'float64')]
    scores = torch.machine.linearScoring2([model1.meanSupervector, model2.meanSupervector], ubm.meanSupervector, ubm.varianceSupervector, [stats1, stats2, stats3], test_channeloffset, True)
    ref_scores = numpy.array([[871.8333333333332, 776.3000000000001, 770.3571428571427], [793.8333333333333, 714.1857142857143, 717.5000000000000]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

    # without normalisation
    scores = torch.machine.linearScoring2([model1.meanSupervector, model2.meanSupervector], ubm.meanSupervector, ubm.varianceSupervector, [stats1, stats2, stats3])
    ref_scores = numpy.array([[2372.9, 5207.7, 5275.7], [2215.7, 4868.1, 4932.1]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

    scores = torch.machine.linearScoring2([model1, model2], ubm, [stats1, stats2, stats3])
    ref_scores = numpy.array([[2372.9, 5207.7, 5275.7], [2215.7, 4868.1, 4932.1]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

    # with normalisation
    scores = torch.machine.linearScoring2([model1.meanSupervector, model2.meanSupervector], ubm.meanSupervector, ubm.varianceSupervector, [stats1, stats2, stats3], True)
    ref_scores = numpy.array( [[790.9666666666667, 743.9571428571428, 753.6714285714285], [738.5666666666667, 695.4428571428572, 704.5857142857144]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())

    scores = torch.machine.linearScoring2([model1, model2], ubm, [stats1, stats2, stats3], True)
    ref_scores = numpy.array( [[790.9666666666667, 743.9571428571428, 753.6714285714285], [738.5666666666667, 695.4428571428572, 704.5857142857144]], 'float64')
    self.assertTrue((abs(scores - ref_scores) < 1e-7).all())
    
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
