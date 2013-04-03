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

"""Tests the WienerMachine
"""

import os, sys
import unittest
import math
import bob
import numpy, numpy.random
import tempfile

class WienerMachineTest(unittest.TestCase):
  """Performs various LinearMachine tests."""

  def test01_initialization(self):

    # Getters/Setters
    m = bob.machine.WienerMachine(5,4,0.5)
    self.assertEqual(m.height, 5)
    self.assertEqual(m.width, 4)
    self.assertEqual(m.shape, (5,4))
    m.height = 8
    m.width = 7
    self.assertEqual(m.height, 8)
    self.assertEqual(m.width, 7)
    self.assertEqual(m.shape, (8,7))
    m.shape = (5,6)
    self.assertEqual(m.height, 5)
    self.assertEqual(m.width, 6)
    self.assertEqual(m.shape, (5,6))
    ps1 = 0.2 + numpy.fabs(numpy.random.randn(5,6))
    ps2 = 0.2 + numpy.fabs(numpy.random.randn(5,6))
    m.ps = ps1
    self.assertTrue( numpy.allclose(m.ps, ps1) )
    m.ps = ps2
    self.assertTrue( numpy.allclose(m.ps, ps2) )
    pn1 = 0.5
    m.pn = pn1
    self.assertTrue( abs(m.pn - pn1) < 1e-5 )
    var_thd = 1e-5
    m.variance_threshold = var_thd
    self.assertTrue( abs(m.variance_threshold - var_thd) < 1e-5 )

    # Comparison operators
    m2 = bob.machine.WienerMachine(m)
    self.assertTrue( m == m2 )
    self.assertFalse( m != m2 )
    m3 = bob.machine.WienerMachine(ps2, pn1)
    m3.variance_threshold = var_thd
    self.assertTrue( m == m3 )
    self.assertFalse( m != m3 )

    # Computation of the Wiener filter W
    w_py = 1 / (1. + m.pn / m.ps)
    self.assertTrue( numpy.allclose(m.w, w_py) )


  def test02_load_save(self):

    m = bob.machine.WienerMachine(5,4,0.5)
    
    # Save and read from file
    filename = str(tempfile.mkstemp(".hdf5")[1])
    m.save(bob.io.HDF5File(filename, 'w'))
    m_loaded = bob.machine.WienerMachine(bob.io.HDF5File(filename))
    self.assertTrue( m == m_loaded )
    self.assertFalse( m != m_loaded )
    self.assertTrue(m.is_similar_to(m_loaded))
    # Make them different
    m_loaded.variance_threshold = 0.001
    self.assertFalse( m == m_loaded )
    self.assertTrue( m != m_loaded )

    # Clean-up
    os.unlink(filename)

  
  def test03_forward(self):

    ps = 0.2 + numpy.fabs(numpy.random.randn(5,6))
    pn = 0.5
    m = bob.machine.WienerMachine(ps,pn)
   
    # Python way
    sample = numpy.random.randn(5,6)
    sample_fft = bob.sp.fft(sample.astype(numpy.complex128))
    w = m.w
    sample_fft_filtered = sample_fft * m.w
    sample_filtered_py = numpy.absolute(bob.sp.ifft(sample_fft_filtered))

    # Bob c++ way
    sample_filtered0 = m.forward(sample) 
    sample_filtered1 = m(sample) 
    sample_filtered2 = numpy.zeros((5,6),numpy.float64)
    m.forward_(sample, sample_filtered2)
    sample_filtered3 = numpy.zeros((5,6),numpy.float64)
    m.forward(sample, sample_filtered3)
    sample_filtered4 = numpy.zeros((5,6),numpy.float64)
    m(sample, sample_filtered4)
    self.assertTrue( numpy.allclose(sample_filtered0, sample_filtered_py) )
    self.assertTrue( numpy.allclose(sample_filtered1, sample_filtered_py) )
    self.assertTrue( numpy.allclose(sample_filtered2, sample_filtered_py) )
    self.assertTrue( numpy.allclose(sample_filtered3, sample_filtered_py) )
    self.assertTrue( numpy.allclose(sample_filtered4, sample_filtered_py) )
