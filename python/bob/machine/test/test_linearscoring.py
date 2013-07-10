#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Wed Jul 13 16:00:04 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

"""Tests on the LinearScoring function
"""

import os, sys
import unittest
import bob
import numpy

class LinearScoringTest(unittest.TestCase):
  """Performs various LinearScoring tests."""

  def test01_LinearScoring(self):
    ubm = bob.machine.GMMMachine(2, 2)
    ubm.weights   = numpy.array([0.5, 0.5], 'float64')
    ubm.means     = numpy.array([[3, 70], [4, 72]], 'float64')
    ubm.variances = numpy.array([[1, 10], [2, 5]], 'float64')
    ubm.variance_thresholds = numpy.array([[0, 0], [0, 0]], 'float64')

    model1 = bob.machine.GMMMachine(2, 2)
    model1.weights   = numpy.array([0.5, 0.5], 'float64')
    model1.means     = numpy.array([[1, 2], [3, 4]], 'float64')
    model1.variances = numpy.array([[9, 10], [11, 12]], 'float64')
    model1.variance_thresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    model2 = bob.machine.GMMMachine(2, 2)
    model2.weights   = numpy.array([0.5, 0.5], 'float64')
    model2.means     = numpy.array([[5, 6], [7, 8]], 'float64')
    model2.variances = numpy.array([[13, 14], [15, 16]], 'float64')
    model2.variance_thresholds = numpy.array([[0, 0], [0, 0]], 'float64')
    
    stats1 = bob.machine.GMMStats(2, 2)
    stats1.sum_px = numpy.array([[1, 2], [3, 4]], 'float64')
    stats1.n = numpy.array([1, 2], 'float64')
    stats1.t = 1+2
    
    stats2 = bob.machine.GMMStats(2, 2)
    stats2.sum_px = numpy.array([[5, 6], [7, 8]], 'float64')
    stats2.n = numpy.array([3, 4], 'float64')
    stats2.t = 3+4
    
    stats3 = bob.machine.GMMStats(2, 2)
    stats3.sum_px = numpy.array([[5, 6], [7, 3]], 'float64')
    stats3.n = numpy.array([3, 4], 'float64')
    stats3.t = 3+4

    test_channeloffset = [numpy.array([9, 8, 7, 6], 'float64'), numpy.array([5, 4, 3, 2], 'float64'), numpy.array([1, 0, 1, 2], 'float64')]

    # Reference scores (from Idiap internal matlab implementation)
    ref_scores_00 = numpy.array([[2372.9, 5207.7, 5275.7], [2215.7, 4868.1, 4932.1]], 'float64')
    ref_scores_01 = numpy.array( [[790.9666666666667, 743.9571428571428, 753.6714285714285], [738.5666666666667, 695.4428571428572, 704.5857142857144]], 'float64')
    ref_scores_10 = numpy.array([[2615.5, 5434.1, 5392.5], [2381.5, 4999.3, 5022.5]], 'float64')
    ref_scores_11 = numpy.array([[871.8333333333332, 776.3000000000001, 770.3571428571427], [793.8333333333333, 714.1857142857143, 717.5000000000000]], 'float64')


    # 1/ Use GMMMachines
    # 1/a/ Without test_channelOffset, without frame-length normalisation
    scores = bob.machine.linear_scoring([model1, model2], ubm, [stats1, stats2, stats3])
    self.assertTrue((abs(scores - ref_scores_00) < 1e-7).all())
    
    # 1/b/ Without test_channelOffset, with frame-length normalisation
    scores = bob.machine.linear_scoring([model1, model2], ubm, [stats1, stats2, stats3], [], True)
    self.assertTrue((abs(scores - ref_scores_01) < 1e-7).all())
    scores = bob.machine.linear_scoring([model1, model2], ubm, [stats1, stats2, stats3], (), True)
    self.assertTrue((abs(scores - ref_scores_01) < 1e-7).all())
    scores = bob.machine.linear_scoring([model1, model2], ubm, [stats1, stats2, stats3], None, True)
    self.assertTrue((abs(scores - ref_scores_01) < 1e-7).all())

    # 1/c/ With test_channelOffset, without frame-length normalisation
    scores = bob.machine.linear_scoring([model1, model2], ubm, [stats1, stats2, stats3], test_channeloffset)
    self.assertTrue((abs(scores - ref_scores_10) < 1e-7).all())

    # 1/d/ With test_channelOffset, with frame-length normalisation
    scores = bob.machine.linear_scoring([model1, model2], ubm, [stats1, stats2, stats3], test_channeloffset, True)
    self.assertTrue((abs(scores - ref_scores_11) < 1e-7).all())

    
    # 2/ Use mean/variance supervectors
    # 2/a/ Without test_channelOffset, without frame-length normalisation
    scores = bob.machine.linear_scoring([model1.mean_supervector, model2.mean_supervector], ubm.mean_supervector, ubm.variance_supervector, [stats1, stats2, stats3])
    self.assertTrue((abs(scores - ref_scores_00) < 1e-7).all())

    # 2/b/ Without test_channelOffset, with frame-length normalisation
    scores = bob.machine.linear_scoring([model1.mean_supervector, model2.mean_supervector], ubm.mean_supervector, ubm.variance_supervector, [stats1, stats2, stats3], [], True)
    self.assertTrue((abs(scores - ref_scores_01) < 1e-7).all())

    # 2/c/ With test_channelOffset, without frame-length normalisation
    scores = bob.machine.linear_scoring([model1.mean_supervector, model2.mean_supervector], ubm.mean_supervector, ubm.variance_supervector, [stats1, stats2, stats3], test_channeloffset)
    self.assertTrue((abs(scores - ref_scores_10) < 1e-7).all())

    # 2/d/ With test_channelOffset, with frame-length normalisation
    scores = bob.machine.linear_scoring([model1.mean_supervector, model2.mean_supervector], ubm.mean_supervector, ubm.variance_supervector, [stats1, stats2, stats3], test_channeloffset, True)
    self.assertTrue((abs(scores - ref_scores_11) < 1e-7).all())

    # 3/ Using single model/sample
    # 3/a/ without frame-length normalisation
    score = bob.machine.linear_scoring(model1.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats1, test_channeloffset[0])
    self.assertTrue(abs(score - ref_scores_10[0,0]) < 1e-7)
    score = bob.machine.linear_scoring(model1.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats2, test_channeloffset[1])
    self.assertTrue(abs(score - ref_scores_10[0,1]) < 1e-7)
    score = bob.machine.linear_scoring(model1.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats3, test_channeloffset[2])
    self.assertTrue(abs(score - ref_scores_10[0,2]) < 1e-7)
    score = bob.machine.linear_scoring(model2.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats1, test_channeloffset[0])
    self.assertTrue(abs(score - ref_scores_10[1,0]) < 1e-7)
    score = bob.machine.linear_scoring(model2.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats2, test_channeloffset[1])
    self.assertTrue(abs(score - ref_scores_10[1,1]) < 1e-7)
    score = bob.machine.linear_scoring(model2.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats3, test_channeloffset[2])
    self.assertTrue(abs(score - ref_scores_10[1,2]) < 1e-7)

    # 3/b/ without frame-length normalisation
    score = bob.machine.linear_scoring(model1.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats1, test_channeloffset[0], True)
    self.assertTrue(abs(score - ref_scores_11[0,0]) < 1e-7)
    score = bob.machine.linear_scoring(model1.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats2, test_channeloffset[1], True)
    self.assertTrue(abs(score - ref_scores_11[0,1]) < 1e-7)
    score = bob.machine.linear_scoring(model1.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats3, test_channeloffset[2], True)
    self.assertTrue(abs(score - ref_scores_11[0,2]) < 1e-7)
    score = bob.machine.linear_scoring(model2.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats1, test_channeloffset[0], True)
    self.assertTrue(abs(score - ref_scores_11[1,0]) < 1e-7)
    score = bob.machine.linear_scoring(model2.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats2, test_channeloffset[1], True)
    self.assertTrue(abs(score - ref_scores_11[1,1]) < 1e-7)
    score = bob.machine.linear_scoring(model2.mean_supervector, ubm.mean_supervector, ubm.variance_supervector, stats3, test_channeloffset[2], True)
    self.assertTrue(abs(score - ref_scores_11[1,2]) < 1e-7)
