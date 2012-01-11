#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <francois.moulin@idiap.ch>
# Thu 14 Apr 15:25:55 2011 

"""Tests histogram computation 
"""

import os, sys
import unittest
import bob
import random
import numpy

def load_gray(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join('data', 'histo', relative_filename)
  array = bob.io.Array(filename)
  return array.get()[0,:,:] 

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
    #bob.io.Array(histo1).save(os.path.join('data', 'histo','image_histo.hdf5'))
    
    histo_ref = bob.io.Array(os.path.join('data', 'histo','image_histo.hdf5')).get()

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
    #bob.io.Array(input_array).save(os.path.join('data', 'histo','input_uint16.hdf5'))
    
    input_array = bob.io.Array(os.path.join('data', 'histo','input_uint16.hdf5')).get()
    
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
    #bob.io.Array(histo1).save(os.path.join('data', 'histo','input_uint16.histo.hdf5'))
    
    histo_ref = bob.io.Array(os.path.join('data', 'histo','input_uint16.histo.hdf5')).get()
    
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
    #bob.io.Array(input_array).save(os.path.join('data', 'histo','input_float.hdf5'))
    
    input_array = bob.io.Array(os.path.join('data', 'histo','input_float.hdf5')).get()
    
    histo2 = numpy.ndarray((10,), 'uint64')
    
    histo1 = bob.ip.histogram(input_array, 0, 1, 10)
    bob.ip.histogram_(input_array, histo2, 0, 1, 10)
    
    # Save computed data
    #bob.io.Array(histo1).save(os.path.join('data', 'histo','input_float.histo.hdf5'))
    
    histo_ref = bob.io.Array(os.path.join('data', 'histo','input_float.histo.hdf5')).get()
    
    self.assertTrue(input_array.size == histo1.sum())
    self.assertTrue(input_array.size == histo2.sum())
    self.assertTrue((histo_ref == histo1).all())
    self.assertTrue((histo_ref == histo2).all())  

  def test04_int32_histoPython(self):
    """Compute the histogram of a int32 random array"""
    
    # Generate random int32 array
    #input_array = numpy.ndarray((50, 70), 'int32')
    #random_int(input_array, -20,20)
    #bob.io.Array(input_array).save(os.path.join('data', 'histo','input_int32.hdf5'))
    
    input_array = bob.io.Array(os.path.join('data', 'histo','input_int32.hdf5')).get()
    
    histo2 = numpy.ndarray((41,), 'uint64')

    histo1 = bob.ip.histogram(input_array, -20, 20, 41)
    bob.ip.histogram_(input_array, histo2, -20, 20, 41)
    
    # Save computed data
    #bob.io.Array(histo).save(os.path.join('data', 'histo','input_int32.histo.hdf5'))
    
    histo_ref = bob.io.Array(os.path.join('data', 'histo','input_int32.histo.hdf5')).get()
    
    self.assertTrue(input_array.size == histo1.sum())
    self.assertTrue(input_array.size == histo2.sum())
    self.assertTrue((histo_ref == histo1).all())
    self.assertTrue((histo_ref == histo2).all())
    
  def test05_uint32_accumulate_histoPython(self):
    """Accumulate the histogram of a int32 random array"""
    
    input_array = bob.io.Array(os.path.join('data', 'histo','input_int32.hdf5')).get()
    
    histo = bob.ip.histogram(input_array, -20, 20, 41)
    
    bob.ip.histogram_(input_array, histo, -20, 20, 41, True)

    histo_ref = bob.io.Array(os.path.join('data', 'histo','input_int32.histo.hdf5')).get()

    self.assertTrue(input_array.size * 2 == histo.sum())
    self.assertTrue((histo_ref * 2 == histo).all())

  def test06_uint16(self):
    """Simple test as described in ticket #101"""
    x = numpy.array([[-1., 1.],[-1., 1.]])
    res = bob.ip.histogram(x, -2, +2, 2)

    histo_ref = numpy.array([2, 2], 'uint64')
    self.assertTrue((histo_ref == res).all())
    
if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()
