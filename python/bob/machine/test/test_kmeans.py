#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
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

"""Tests the KMeans machine
"""

import os, sys
import unittest
import bob
import numpy, math
import tempfile

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon)
  
class KMeansMachineTest(unittest.TestCase):
  """Performs various KMeans machine-related tests."""

  def test01_KMeansMachine(self):
    """Test a KMeansMachine"""

    means = numpy.array([[3, 70, 0], [4, 72, 0]], 'float64')
    mean  = numpy.array([3,70,1], 'float64')

    # Initializes a KMeansMachine
    km = bob.machine.KMeansMachine(2,3)
    km.means = means
    self.assertTrue( km.dim_c == 2 )
    self.assertTrue( km.dim_d == 3 )

    # Sets and gets
    self.assertTrue( (km.means == means).all() )
    self.assertTrue( (km.get_mean(0) == means[0,:]).all() )
    self.assertTrue( (km.get_mean(1) == means[1,:]).all() )
    km.set_mean(0, mean)
    self.assertTrue( (km.get_mean(0) == mean).all() )

    # Distance and closest mean
    eps = 1e-10
    self.assertTrue( equals( km.get_distance_from_mean(mean, 0), 0, eps) )
    self.assertTrue( equals( km.get_distance_from_mean(mean, 1), 6, eps) )
    (index, dist) = km.get_closest_mean(mean)
    self.assertTrue( index == 0)
    self.assertTrue( equals( dist, 0, eps) )
    self.assertTrue( equals( km.get_min_distance(mean), 0, eps) )

    # Loads and saves
    filename = str(tempfile.mkstemp(".hdf5")[1])
    km.save(bob.io.HDF5File(filename, 'w'))
    km_loaded = bob.machine.KMeansMachine(bob.io.HDF5File(filename))
    self.assertTrue( km == km_loaded )

    # Resize 
    km.resize(4,5)
    self.assertTrue( km.dim_c == 4 )
    self.assertTrue( km.dim_d == 5 )

    # Copy constructor and comparison operators
    km.resize(2,3)
    km2 = bob.machine.KMeansMachine(km)
    self.assertTrue( km2 == km)
    self.assertFalse( km2 != km)
    means2 = numpy.array([[3, 70, 0], [4, 72, 2]], 'float64')
    km2.means = means2
    self.assertFalse( km2 == km)
    self.assertTrue( km2 != km)

    # Clean-up
    os.unlink(filename)
