#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Sat Sep 1 9:43:00 2012 +0200
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

"""Test trainer package
"""

import os, sys
import unittest
import bob
import random
import numpy

class LLRTest(unittest.TestCase):
  """Performs various tests for Linear Logistic Regression."""
  
  def test01_llr(self):

    # Tests our LLR Trainer.
    ar1 = numpy.array([
      [1.,1.2,-1.],
      [2.,2.1,2.2],
      [3.,2.9,3.1],
      [4.,3.7,4.],
      [5.,5.2,4.9],
      [6.,6.1,5.9],
      [7.,7.,7.3],
      ], dtype='float64')

    ar2 = numpy.array([
      [-10.,-9.2,-1.],
      [-5.,-4.1,-0.5],
      [-10.,-9.9,-1.8],
      [-5.,-5.4,-0.3],
      [-10.,-9.3,-0.7],
      [-5.,-4.5,-0.5],
      [-10.,-9.7,-1.2],
      [-5.,-4.8,-0.2],
      ], dtype='float64')

    # Expected trained machine
    #weights_ref= numpy.array([[13.5714], [19.3997], [-0.6432]])
    weights_ref= numpy.array([[1.75536], [2.69297], [-0.54142]])
    #bias_ref = numpy.array([55.3255])
    bias_ref = numpy.array([7.26999])

    # Features and expected outputs of the trained machine
    feat1 = numpy.array([1.,2.,3.])
    #out1 = 105.7668
    out1 = 12.78703
    feat2 = numpy.array([2.,3.,4.])
    #out2 = 138.0947
    out2 = 16.69394

  
    # Trains a machine (method 1)
    T = bob.trainer.LLRTrainer(0.5, 1e-5, 30)
    machine1 = T.train(ar1,ar2)

    # Makes sure results are good
    print machine1.weights
    print machine1.biases
    self.assertTrue( (abs(machine1.weights - weights_ref) < 2e-4).all() )
    self.assertTrue( (abs(machine1.biases - bias_ref) < 2e-4).all() )
    self.assertTrue( abs(machine1(feat1) - out1) < 2e-4 )
    self.assertTrue( abs(machine1(feat2) - out2) < 2e-4 )

    # Trains a machine (method 2)
    machine2 = bob.machine.LinearMachine()
    T.train(machine2, ar1, ar2)

    # Makes sure results are good
    self.assertTrue( (abs(machine2.weights - weights_ref) < 2e-4).all() )
    self.assertTrue( (abs(machine2.biases - bias_ref) < 2e-4).all() )
    self.assertTrue( abs(machine2(feat1) - out1) < 2e-4 )
    self.assertTrue( abs(machine2(feat2) - out2) < 2e-4 )
