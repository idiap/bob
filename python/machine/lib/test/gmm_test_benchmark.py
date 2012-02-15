#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Niklas Johansson <niklas.johansson@idiap.ch>
# Wed May 11 15:00:05 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

"""Tests machine package
"""

import os, sys
import unittest
import numpy
import bob
import itertools

def my_strip(str):
  return str.strip('\n')

def list_of_lines(lines):
  vals = []
  for line in lines:
    vals.append(map(float, line.rsplit()))
      
  return vals

def read_lightsaber(filename):

  f = open(filename, 'r')
  v = f.readlines()
  f.close()
  v = map(my_strip, v)

  weis  = [v[i] for i in range(0, len(v), 3)]
  means = [v[i] for i in range(1, len(v), 3)]
  varis = [v[i] for i in range(2, len(v), 3)]

  weis  = map(float, weis)
  means = list_of_lines(means)
  varis = list_of_lines(varis)

  return weis, means, varis

class MachineTest(unittest.TestCase):
  """Performs various machine tests."""

  def test01_GMMMachine(self):
    """Test a GMMMachine"""

    sampler = bob.trainer.SimpleFrameSampler(bob.io.Arrayset("1028_m_g1_s03_1028_en_2.bindata"))
    gmm = bob.machine.GMMMachine(3, 91)

    weis, means, varis = read_lightsaber("tan-triggs-64x80-normalised-D91-gmm512-B24-BANCA.data.s")
    print means

    gmm.weights   = numpy.array(weis, 'float64')
    gmm.means     = numpy.array(means, 'float64')
    gmm.variances = numpy.array(varis, 'float64')

    gmm.print_()

    gmm.forward(sampler)

    stats = bob.machine.GMMStats(3, 91)
    gmm.accStatistics(sampler, stats)
    stats.print_()

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(MachineTest)
