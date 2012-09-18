#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue May 31 16:55:10 2011 +0200
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

"""Tests on the LinearMachine infrastructure.
"""

import os, sys
import unittest
import math
import bob
import numpy
import pkg_resources

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

MACHINE = F('linear-test.hdf5')

class MachineTest(unittest.TestCase):
  """Performs various LinearMachine tests."""

  def test01_Initialization(self):

    # Two inputs and 1 output
    m = bob.machine.LinearMachine(2,1)
    self.assertTrue( (m.weights == 0.0).all() )
    self.assertEqual( m.weights.shape, (2,1) )
    self.assertTrue( (m.biases == 0.0).all() )
    self.assertEqual( m.biases.shape, (1,) )

    # Start by providing the data
    w = numpy.array([[0.4, 0.1], [0.4, 0.2], [0.2, 0.7]], 'float64')
    m = bob.machine.LinearMachine(w)
    b = numpy.array([0.3, -3.0], 'float64')
    isub = numpy.array([0., 0.5, 0.5], 'float64')
    idiv = numpy.array([0.5, 1.0, 1.0], 'float64')
    m.input_subtract = isub
    m.input_divide = idiv
    m.biases = b
    m.activation = bob.machine.Activation.TANH

    self.assertTrue( (m.input_subtract == isub).all() )
    self.assertTrue( (m.input_divide == idiv).all() )
    self.assertTrue( (m.weights == w).all() )
    self.assertTrue( (m.biases == b). all() )
    self.assertEqual(m.activation, bob.machine.Activation.TANH)
    # Save to file
    # c = bob.io.HDF5File("bla.hdf5", 'w')
    # m.save(c)

    # Start by reading data from a file
    c = bob.io.HDF5File(MACHINE)
    m = bob.machine.LinearMachine(c)
    self.assertTrue( (m.weights == w).all() )
    self.assertTrue( (m.biases == b). all() )

    # Makes sure we cannot stuff incompatible data
    w = numpy.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
    m = bob.machine.LinearMachine(w)
    b = numpy.array([0.3, -3.0, 2.7, -18, 52], 'float64') #wrong
    self.assertRaises(RuntimeError, setattr, m, 'biases', b)
    self.assertRaises(RuntimeError, setattr, m, 'input_subtract', b)
    self.assertRaises(RuntimeError, setattr, m, 'input_divide', b)

  def test02_Correctness(self):

    # Tests the correctness of a linear machine
    c = bob.io.HDF5File(MACHINE)
    m = bob.machine.LinearMachine(c)

    def presumed(ivalue):
      """Calculates, by hand, the presumed output given the input"""

      # These are the supposed preloaded values from the file "MACHINE"
      isub = numpy.array([0., 0.5, 0.5], 'float64')
      idiv = numpy.array([0.5, 1.0, 1.0], 'float64')
      w = numpy.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
      b = numpy.array([0.3, -3.0], 'float64')
      act = math.tanh
  
      return numpy.array([ act((w[i,:]*((ivalue-isub)/idiv)).sum() + b[i]) for i in range(w.shape[0]) ], 'float64')

    testing = [
        [1,1,1],
        [0.5,0.2,200],
        [-27,35.77,0],
        [12,0,0],
        ]

    # 1D case
    maxerr = numpy.ndarray((2,), 'float64')
    maxerr.fill(1e-10)
    for k in testing:
      input = numpy.array(k, 'float64')
      self.assertTrue ( (abs(presumed(input) - m(input)) < maxerr).all() )

    # 2D case
    output = m(testing)
    for i, k in enumerate(testing):
      input = numpy.array(k, 'float64')
      self.assertTrue ( (abs(presumed(input) - output[i,:]) < maxerr).all() )

  def test03_UserAllocation(self):

    # Tests the correctness of a linear machine
    c = bob.io.HDF5File(MACHINE)
    m = bob.machine.LinearMachine(c)

    def presumed(ivalue):
      """Calculates, by hand, the presumed output given the input"""

      # These are the supposed preloaded values from the file "MACHINE"
      isub = numpy.array([0., 0.5, 0.5], 'float64')
      idiv = numpy.array([0.5, 1.0, 1.0], 'float64')
      w = numpy.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
      b = numpy.array([0.3, -3.0], 'float64')
      act = math.tanh
  
      return numpy.array([ act((w[i,:]*((ivalue-isub)/idiv)).sum() + b[i]) for i in range(w.shape[0]) ], 'float64')

    testing = [
        [1,1,1],
        [0.5,0.2,200],
        [-27,35.77,0],
        [12,0,0],
        ]

    # 1D case
    maxerr = numpy.ndarray((2,), 'float64')
    maxerr.fill(1e-10)
    output = numpy.ndarray((2,), 'float64')
    for k in testing:
      input = numpy.array(k, 'float64')
      m(input, output)
      self.assertTrue ( (abs(presumed(input) - output) < maxerr).all() )

    # 2D case
    output = numpy.ndarray((len(testing), 2), 'float64')
    m(testing, output)
    for i, k in enumerate(testing):
      input = numpy.array(k, 'float64')
      self.assertTrue ( (abs(presumed(input) - output[i,:]) < maxerr).all() )

  def test04_comparisons(self):

    # Start by creating the data
    w1 = numpy.array([[0.4, 0.1], [0.4, 0.2], [0.2, 0.7]], 'float64')
    w2 = numpy.array([[0.4, 1.1], [0.4, 0.2], [0.2, 0.7]], 'float64')
    b1 = numpy.array([0.3, -3.0], 'float64')
    b2 = numpy.array([0.3, 3.0], 'float64')
    isub1 = numpy.array([0., 0.5, 0.5], 'float64')
    isub2 = numpy.array([0.5, 0.5, 0.5], 'float64')
    idiv1 = numpy.array([0.5, 1.0, 1.0], 'float64')
    idiv2 = numpy.array([1.5, 1.0, 1.0], 'float64')

    # Creates LinearMachine's
    m1 = bob.machine.LinearMachine(w1)
    m1.input_subtract = isub1
    m1.input_divide = idiv1
    m1.biases = b1
    m1.activation = bob.machine.Activation.TANH

    m1b = bob.machine.LinearMachine(m1)
    m1c = bob.machine.LinearMachine(w1)
    m1c.input_subtract = isub1
    m1c.input_divide = idiv1
    m1c.biases = b1
    m1c.activation = bob.machine.Activation.TANH

    m2 = bob.machine.LinearMachine(w2)
    m2.input_subtract = isub1
    m2.input_divide = idiv1
    m2.biases = b1
    m2.activation = bob.machine.Activation.TANH

    m3 = bob.machine.LinearMachine(w1)
    m3.input_subtract = isub2
    m3.input_divide = idiv1
    m3.biases = b1
    m3.activation = bob.machine.Activation.TANH

    m4 = bob.machine.LinearMachine(w1)
    m4.input_subtract = isub1
    m4.input_divide = idiv2
    m4.biases = b1
    m4.activation = bob.machine.Activation.TANH

    m5 = bob.machine.LinearMachine(w1)
    m5.input_subtract = isub1
    m5.input_divide = idiv1
    m5.biases = b2
    m5.activation = bob.machine.Activation.TANH

    m6 = bob.machine.LinearMachine(w1)
    m6.input_subtract = isub1
    m6.input_divide = idiv1
    m6.biases = b1
    m6.activation = bob.machine.Activation.LINEAR

    # Compares them using the overloaded operators == and !=
    self.assertTrue( m1 == m1b )
    self.assertFalse( m1 != m1b )
    self.assertTrue( m1 == m1c )
    self.assertFalse( m1 != m1c )
    self.assertFalse( m1 == m2 )
    self.assertTrue( m1 != m2 )
    self.assertFalse( m1 == m3 )
    self.assertTrue( m1 != m3 )
    self.assertFalse( m1 == m4 )
    self.assertTrue( m1 != m4 )
    self.assertFalse( m1 == m5 )
    self.assertTrue( m1 != m5 )
    self.assertFalse( m1 == m6 )
    self.assertTrue( m1 != m6 )

