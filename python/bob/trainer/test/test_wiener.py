#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

"""Tests the WienerTrainer
"""

import os, sys
import unittest
import math
import bob
import numpy, numpy.random
import tempfile

def train_wiener_ps(training_set):

  # Python implementation
  n_samples = training_set.shape[0]
  height = training_set.shape[1]
  width = training_set.shape[2]
  training_fftabs = numpy.zeros((n_samples, height, width), dtype=numpy.float64)

  for n in range(n_samples):
    sample = (training_set[n,:,:]).astype(numpy.complex128)
    training_fftabs[n,:,:] = numpy.absolute(bob.sp.fft(sample))

  mean = numpy.mean(training_fftabs, axis=0)

  for n in range(n_samples):
    training_fftabs[n,:,:] -= mean

  training_fftabs = training_fftabs * training_fftabs
  var_ps = numpy.mean(training_fftabs, axis=0)

  return var_ps


class WienerTrainerTest(unittest.TestCase):
  """Performs various WienerTrainer tests."""

  def test01_initialization(self):

    # Constructors and comparison operators
    t1 = bob.trainer.WienerTrainer()
    t2 = bob.trainer.WienerTrainer()
    t3 = bob.trainer.WienerTrainer(t2)
    t4 = t3
    self.assertTrue( t1 == t2)
    self.assertFalse( t1 != t2)
    self.assertTrue( t1.is_similar_to(t2) )
    self.assertTrue( t1 == t3)
    self.assertFalse( t1 != t3)
    self.assertTrue( t1.is_similar_to(t3) )
    self.assertTrue( t1 == t4)
    self.assertFalse( t1 != t4)
    self.assertTrue( t1.is_similar_to(t4) )


  def test02_train(self):

    n_samples = 20
    height = 5
    width = 6
    training_set = 0.2 + numpy.fabs(numpy.random.randn(n_samples, height, width))

    # Python implementation
    var_ps = train_wiener_ps(training_set)
    # Bob C++ implementation (variant 1) + comparison against python one
    t = bob.trainer.WienerTrainer()
    m1 = t.train(training_set)
    self.assertTrue( numpy.allclose(var_ps, m1.ps) )
    # Bob C++ implementation (variant 2) + comparison against python one
    m2 = bob.machine.WienerMachine(height, width, 0.)
    t.train(m2, training_set)
    self.assertTrue( numpy.allclose(var_ps, m2.ps) )

