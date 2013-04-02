#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Apr 2 21:40:0 2013 +0200
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

"""Test the whitening trainer
"""

import os, sys
import unittest
import bob
import numpy

class WhiteningTest(unittest.TestCase):
  """Performs various trainer tests for the LinearMachine."""
  
  def test01_whitening(self):

    # Tests our SVD/PCA extractor.
    data = numpy.array([[ 1.2622, -1.6443, 0.1889],
                        [ 0.4286, -0.8922, 1.3020],
                        [-0.6613,  0.0430, 0.6377],
                        [-0.8718, -0.4788, 0.3988],
                        [-0.0098, -0.3121,-0.1807],
                        [ 0.4301,  0.4886, -0.1456]])
    sample = numpy.array([1, 2, 3.])

    # Expected results (from matlab)
    mean_ref = numpy.array([0.096324163333333, -0.465965438333333, 0.366839091666667])
    whit_ref = numpy.array([[1.608410253685985,                  0,                  0],
                            [1.079813355720326,  1.411083365535711,                  0],
                            [0.693459921529905,  0.571417184139332,  1.800117179839927]])
    sample_whitened_ref = numpy.array([5.942255453628436, 4.984316201643742, 4.739998188373740])
    
    # Runs whitening
    t = bob.trainer.WhiteningTrainer()
    m = bob.machine.LinearMachine(3,3)
    t.train(m, data)
    s = m.forward(sample)

    # Makes sure results are good
    eps = 1e-4
    self.assertTrue( numpy.allclose(m.input_subtract, mean_ref, eps, eps) )
    self.assertTrue( numpy.allclose(m.weights, whit_ref, eps, eps) )
    self.assertTrue( numpy.allclose(s, sample_whitened_ref, eps, eps) )

