#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Thu Jun 14 14:45:06 CEST 2012
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

"""Test BIC trainer and machine
"""

import os, sys
import unittest
import bob
import numpy

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class BICTrainerAndMachineTest(unittest.TestCase):
  """Performs various BIC trainer and machine tests."""
  
  def training_data(self):
    data = ((10., 4., 6., 8., 2.),
            (8., 2., 4., 6., 0.),
            (12., 6., 8., 10., 4.),
            (11., 3., 7., 7., 3.),
            (9., 5., 5., 9., 1.))
    
    intra_data = bob.io.Arrayset()
    extra_data = bob.io.Arrayset()
    for i in range(5):
      intra_data.append(numpy.array(data[i], dtype=numpy.float64))
      extra_data.append(numpy.array(data[i], dtype=numpy.float64) * -1.)
      
    return (intra_data, extra_data)
  
  def eval_data(self, which):
    eval_data = numpy.ndarray((5,), dtype=numpy.float64)
    if which == 0:
      eval_data.fill(0.)
    elif which == 1:
      eval_data.fill(10.)
      
    return eval_data
    
  def test_IEC(self):
    """Tests the IEC training of the BICTrainer."""
    intra_data, extra_data = self.training_data()
    
    # train BIC machine
    machine = bob.machine.BICMachine()
    trainer = bob.trainer.BICTrainer()

    # train machine with intrapersonal data only
    trainer.train(machine, intra_data, intra_data)
    # => every result should be zero
    self.assertAlmostEqual(machine(self.eval_data(0)), 0.)
    self.assertAlmostEqual(machine(self.eval_data(1)), 0.)
    
    # re-train the machine with intra- and extrapersonal data
    trainer.train(machine, intra_data, extra_data)
    # now, only the input vector 0 should give log-likelihood 0
    self.assertAlmostEqual(machine(self.eval_data(0)), 0.)
    # while a positive vector should give a positive result
    self.assertTrue(machine(self.eval_data(1)) > 0.)
    
  def test_BIC(self):
    """Tests the BIC training of the BICTrainer."""
    intra_data, extra_data = self.training_data()
    
    # train BIC machine
    trainer = bob.trainer.BICTrainer(2,2)
    
    # The data are chosen such that the third eigenvalue is zero. 
    # Hence, calculating rho (i.e., using the Distance From Feature Space) is impossible
    machine = bob.machine.BICMachine(True)
    def should_raise():
      trainer.train(machine, intra_data, intra_data)
    self.assertRaises(ZeroDivisionError, should_raise)

    # So, now without rho...
    machine = bob.machine.BICMachine(False)
    
    # First, train the machine with intrapersonal data only
    trainer.train(machine, intra_data, intra_data)
    
    # => every result should be zero
    self.assertAlmostEqual(machine(self.eval_data(0)), 0.)
    self.assertAlmostEqual(machine(self.eval_data(1)), 0.)
    
    # re-train the machine with intra- and extrapersonal data
    trainer.train(machine, intra_data, extra_data)
    # now, only the input vector 0 should give log-likelihood 0
    self.assertAlmostEqual(machine(self.eval_data(0)), 0.)
    # while a positive vector should give a positive result
    self.assertTrue(machine(self.eval_data(1)) > 0.)
                           

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(BICTrainerAndMachineTest)
