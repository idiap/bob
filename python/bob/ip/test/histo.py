#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Mon Apr 18 16:08:34 2011 +0200
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

"""Tests histogram computation 
"""

import os, sys
import unittest
import bob
import random
import numpy
import pkg_resources

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

def load_gray(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join('histo', relative_filename)
  array = bob.io.load(F(filename))
  return array[0,:,:] 

def random_int(array, min_value, max_value):
  for i in range(0,array.shape()[0]):
    for j in range(0,array.shape()[1]):
      array[i, j] = random.randint(min_value, max_value)
        
def random_float(array, min_value, max_value):
  for i in range(0,array.shape()[0]):
    for j in range(0,array.shape()[1]):
      array[i, j] = random.uniform(min_value, max_value) 


class HistogramTest(unittest.TestCase):
  """Performs various histogram tests."""

  def test01_uint8_histoPython(self):
    """Compute the histogram of a uint8 image"""
    input_image = load_gray('image.ppm')
    

    histo1 = bob.ip.histogram(input_image)
    histo2 = bob.ip.histogram(input_image, 255)
    histo3 = bob.ip.histogram(input_image, 0, 255)
    histo4 = bob.ip.histogram(input_image, 0, 255, 256)
    
    histo5 = numpy.ndarray((256,), 'uint64')
    histo6 = numpy.ndarray((256,), 'uint64')
    histo7 = numpy.ndarray((256,), 'uint64')
    histo8 = numpy.ndarray((256,), 'uint64')
    
    bob.ip.histogram_(input_image, histo5)
    bob.ip.histogram_(input_image, histo6, 255)
    bob.ip.histogram_(input_image, histo7, 0, 255)
    bob.ip.histogram_(input_image, histo8, 0, 255, 256)
    
    # Save the computed data
    #bob.io.save(histo1, os.path.join('histo','image_histo.hdf5'))
    
    histo_ref = bob.io.load(F(os.path.join('histo','image_histo.hdf5')))

    self.assertTrue(input_image.size == histo1.sum())
    self.assertTrue(input_image.size == histo2.sum())
    self.assertTrue(input_image.size == histo3.sum())
    self.assertTrue(input_image.size == histo4.sum())
    self.assertTrue(input_image.size == histo5.sum())
    self.assertTrue(input_image.size == histo6.sum())
    self.assertTrue(input_image.size == histo7.sum())
    self.assertTrue(input_image.size == histo8.sum())
    self.assertTrue((histo_ref == histo1).all())
    self.assertTrue((histo_ref == histo2).all())
    self.assertTrue((histo_ref == histo3).all())
    self.assertTrue((histo_ref == histo4).all())
    self.assertTrue((histo_ref == histo5).all())
    self.assertTrue((histo_ref == histo6).all())
    self.assertTrue((histo_ref == histo7).all())
    self.assertTrue((histo_ref == histo8).all())
    
  def test02_uint16_histoPython(self):
    """Compute the histogram of a uint16 random array"""
    
    # Generate random uint16 array
    #input_array = numpy.ndarray((50, 70), 'uint16')
    #random_int(input_array, 0, 65535)
    #bob.io.save(input_array, os.path.join('histo','input_uint16.hdf5'))
    
    input_array = bob.io.load(F(os.path.join('histo','input_uint16.hdf5')))
    
    histo1 = bob.ip.histogram(input_array)
    histo2 = bob.ip.histogram(input_array, 65535)
    histo3 = bob.ip.histogram(input_array, 0, 65535)
    histo4 = bob.ip.histogram(input_array, 0, 65535, 65536)
    
    histo5 = numpy.ndarray((65536,), 'uint64')
    histo6 = numpy.ndarray((65536,), 'uint64')
    histo7 = numpy.ndarray((65536,), 'uint64')
    histo8 = numpy.ndarray((65536,), 'uint64')
    
    bob.ip.histogram_(input_array, histo5)
    bob.ip.histogram_(input_array, histo6, 65535)
    bob.ip.histogram_(input_array, histo7, 0, 65535)
    bob.ip.histogram_(input_array, histo8, 0, 65535, 65536)
    
    # Save computed data
    #bob.io.save(histo1, os.path.join('histo','input_uint16.histo.hdf5'))
    
    histo_ref = bob.io.load(F(os.path.join('histo','input_uint16.histo.hdf5')))
    
    self.assertTrue(input_array.size == histo1.sum())
    self.assertTrue(input_array.size == histo2.sum())
    self.assertTrue(input_array.size == histo3.sum())
    self.assertTrue(input_array.size == histo4.sum())
    self.assertTrue(input_array.size == histo5.sum())
    self.assertTrue(input_array.size == histo6.sum())
    self.assertTrue(input_array.size == histo7.sum())
    self.assertTrue(input_array.size == histo8.sum())
    self.assertTrue((histo_ref == histo1).all())
    self.assertTrue((histo_ref == histo2).all())
    self.assertTrue((histo_ref == histo3).all())
    self.assertTrue((histo_ref == histo4).all())
    self.assertTrue((histo_ref == histo5).all())
    self.assertTrue((histo_ref == histo6).all())
    self.assertTrue((histo_ref == histo7).all())
    self.assertTrue((histo_ref == histo8).all())
    
  def test03_float_histoPython(self):
    """Compute the histogram of a float random array"""
    
    # Generate random float32 array
    #input_array = numpy.ndarray((50, 70), 'float32')
    #random_float(input_array, 0, 1)
    #bob.io.save(input_array, os.path.join('histo','input_float.hdf5'))
    
    input_array = bob.io.load(F(os.path.join('histo','input_float.hdf5')))
    
    histo2 = numpy.ndarray((10,), 'uint64')
    
    histo1 = bob.ip.histogram(input_array, 0, 1, 10)
    bob.ip.histogram_(input_array, histo2, 0, 1, 10)
    
    # Save computed data
    #bob.io.save(histo1,os.path.join('histo','input_float.histo.hdf5'))
    
    histo_ref = bob.io.load(F(os.path.join('histo','input_float.histo.hdf5')))
    
    self.assertTrue(input_array.size == histo1.sum())
    self.assertTrue(input_array.size == histo2.sum())
    self.assertTrue((histo_ref == histo1).all())
    self.assertTrue((histo_ref == histo2).all())  

  def test04_int32_histoPython(self):
    """Compute the histogram of a int32 random array"""
    
    # Generate random int32 array
    #input_array = numpy.ndarray((50, 70), 'int32')
    #random_int(input_array, -20,20)
    #bob.io.save(input_array,os.path.join('histo','input_int32.hdf5'))
    
    input_array = bob.io.load(F(os.path.join('histo','input_int32.hdf5')))
    
    histo2 = numpy.ndarray((41,), 'uint64')

    histo1 = bob.ip.histogram(input_array, -20, 20, 41)
    bob.ip.histogram_(input_array, histo2, -20, 20, 41)
    
    # Save computed data
    #bob.io.save(histo, os.path.join('histo','input_int32.histo.hdf5'))
    
    histo_ref = bob.io.load(F(os.path.join('histo','input_int32.histo.hdf5')))
    
    self.assertTrue(input_array.size == histo1.sum())
    self.assertTrue(input_array.size == histo2.sum())
    self.assertTrue((histo_ref == histo1).all())
    self.assertTrue((histo_ref == histo2).all())
    
  def test05_uint32_accumulate_histoPython(self):
    """Accumulate the histogram of a int32 random array"""
    
    input_array = bob.io.load(F(os.path.join('histo','input_int32.hdf5')))
    
    histo = bob.ip.histogram(input_array, -20, 20, 41)
    
    bob.ip.histogram_(input_array, histo, -20, 20, 41, True)

    histo_ref = bob.io.load(F(os.path.join('histo','input_int32.histo.hdf5')))

    self.assertTrue(input_array.size * 2 == histo.sum())
    self.assertTrue((histo_ref * 2 == histo).all())

  def test06_uint16(self):
    """Simple test as described in ticket #101"""
    x = numpy.array([[-1., 1.],[-1., 1.]])
    res = bob.ip.histogram(x, -2, +2, 2)

    histo_ref = numpy.array([2, 2], 'uint64')
    self.assertTrue((histo_ref == res).all())
