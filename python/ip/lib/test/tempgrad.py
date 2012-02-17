#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Sep 7 08:52:45 2011 +0200
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


"""Tests our Temporal Gradient utilities going through some example data.
"""

import os, sys
import unittest
import bob
import math
import numpy
import scipy.signal

def load_gray(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join("flow", relative_filename)
  array = bob.io.Array(filename)
  return array.get()[0,:,:] 

def load_known_flow(relative_filename):
  filename = os.path.join("flow", relative_filename)
  array = bob.io.Array(filename)
  data = array.get()
  return data[:,:,0].astype('float64'), data[:,:,1].astype('float64')

def make_image_pair_1():
  """Creates two images for you to calculate the flow
  
  1 1 => 1 1
  1 1    1 2
  
  """
  im1 = numpy.ones((2,2), 'uint8')
  im2 = im1.copy()
  im2[1,1] = 2
  return im1, im2

def make_image_pair_2():
  """Creates two images for you to calculate the flow
  
  10 10 10 10 10    10 10 10 10 10
  10  5  5  5  5    10 10 10 10 10
  10  5  5  5  5 => 10 10  5  5  5
  10 10 10 10 10    10 10  5  5  5
  10 10 10 10 10    10 10 10 10 10
  
  """
  im1 = 10 * numpy.ones((5,5), 'uint8')
  im1[1:3, 1:] = 5
  im2 = 10 * numpy.ones((5,5), 'uint8')
  im2[2:4, 2:] = 5
  return im1, im2

def make_image_tripplet_1():
  """Creates two images for you to calculate the flow
  
  10 10 10 10 10    10 10 10 10 10    10 10 10 10 10
  10  5  5  5  5    10 10 10 10 10    10 10 10 10 10
  10  5  5  5  5 => 10 10  5  5  5 => 10 10 10 10 10
  10 10 10 10 10    10 10  5  5  5    10 10 10  5  5
  10 10 10 10 10    10 10 10 10 10    10 10 10  5  5
  
  """
  im1 = 10* numpy.ones((5,5), 'uint8')
  im1[1:3, 1:] = 5
  im2 = 10* numpy.ones((5,5), 'uint8')
  im2[2:4, 2:] = 5
  im3 = 10* numpy.ones((5,5), 'uint8')
  im3[3:, 3:] = 5
  return im1, im2, im3

def Forward_Ex(im1, im2):
  """Calculates the approximate forward derivative in X direction"""
  e = numpy.ndarray(im1.shape, 'float64')
  e.fill(0) #only last column should keep this value
  im1 = im1.astype('float64')
  im2 = im2.astype('float64')
  for i in range(im1.shape[0]-1):
    for j in range(im1.shape[1]-1):
      e[i,j] = 0.25 * ( im1[i,j+1] - im1[i,j] +
                        im1[i+1,j+1] - im1[i+1,j] +
                        im2[i,j+1] - im2[i,j] +
                        im2[i+1,j+1] - im2[i+1,j] )
  for j in range(im1.shape[1]-1): #last row there is no i+1
    e[-1,j] = 0.5 * ( im1[-1,j+1]-im1[-1,j]+im2[-1,j+1]-im2[-1,j] )
  return e

def Forward_Ey(im1, im2):
  """Calculates the approximate forward derivative in Y direction"""
  e = numpy.ndarray(im1.shape, 'float64')
  e.fill(0) #only last row should keep this value
  im1 = im1.astype('float64')
  im2 = im2.astype('float64')
  for i in range(im1.shape[0]-1):
    for j in range(im1.shape[1]-1):
      e[i,j] = 0.25 * ( im1[i+1,j] - im1[i,j] +
                        im1[i+1,j+1] - im1[i,j+1] +
                        im2[i+1,j] - im2[i,j] +
                        im2[i+1,j+1] - im2[i,j+1] )
  for i in range(im1.shape[0]-1): #last column there is no j+1
    e[i,-1] = 0.5 * ( im1[i+1,-1]-im1[i,-1]+im2[i+1,-1]-im2[i,-1] )
  return e

def Forward_Et(im1, im2):
  """Calculates the approximate derivative in T (time) direction"""
  e = numpy.ndarray(im1.shape, 'float64')
  e.fill(0) #only last row should keep this value
  im1 = im1.astype('float64')
  im2 = im2.astype('float64')
  for i in range(im1.shape[0]-1):
    for j in range(im1.shape[1]-1):
      e[i,j] = 0.25 * ( im2[i,j] - im1[i,j] +
                        im2[i+1,j] - im1[i+1,j] +
                        im2[i,j+1] - im1[i,j+1] +
                        im2[i+1,j+1] - im1[i+1,j+1] )
  for i in range(im1.shape[0]-1): #last column there is no j+1
    e[i,-1] = 0.5 * ( im2[i,-1] - im1[i,-1] + im2[i+1,-1] - im1[i+1,-1] )
  for j in range(im1.shape[1]-1): #last row there is no i+1
    e[-1,j] = 0.5 * ( im2[-1,j] - im1[-1,j] + im2[-1,j+1] - im1[-1,j+1] )
  e[-1, -1] = im2[-1,-1] - im1[-1,-1]
  return e

def LaplacianBorder(u):
  """Calculates the Laplacian border estimate"""
  result = numpy.ndarray(u.shape, 'float64')
  for i in range(1, u.shape[0]-1): #middle of the image
    for j in range(1, u.shape[1]-1):
      result[i,j] = 0.25 * ( 4*u[i,j] - u[i-1,j] - u[i,j+1] - u[i+1,j] - u[i,j-1] )

  #middle of border rows
  for j in range(1, u.shape[1]-1): #first row (i-1) => not bound
    result[0,j] = 0.25 * ( 3*u[0,j] - u[0,j+1] - u[1,j] - u[0,j-1] ) 
  for j in range(1, u.shape[1]-1): #last row (i+1) => not bound
    result[-1,j] = 0.25 * ( 3*u[-1,j] - u[-1,j+1] - u[-2,j] - u[-1,j-1] )
  #middle of border columns
  for i in range(1, u.shape[0]-1): #first column (j-1) => not bound
    result[i,0] = 0.25 * ( 3*u[i,0] - u[i-1,0] - u[i+1,0] - u[i,1] ) 
  for i in range(1, u.shape[0]-1): #last column (j+1) => not bound
    result[i,-1] = 0.25 * ( 3*u[i,-1] - u[i-1,-1] - u[i+1,-1] - u[i,-2] ) 

  #corner pixels
  result[0,0] = 0.25 * ( 2*u[0,0] - u[0,1] - u[1,0] )
  result[0,-1] = 0.25 * ( 2*u[0,-1] - u[0,-2] - u[1,-1] )
  result[-1,0] = 0.25 * ( 2*u[-1,0] - u[-2,0] - u[-1,1] )
  result[-1,-1] = 0.25 * ( 2*u[-1,-1] - u[-2,-1] - u[-1,-2] )

  return result

def Central_Ex(im1, im2, im3):
  """Calculates the approximate central derivative in X direction"""

  Kx = numpy.array([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]], 'float64')

  c1 = numpy.ndarray(im1.shape, 'float64')
  c2 = numpy.ndarray(im2.shape, 'float64')
  c3 = numpy.ndarray(im3.shape, 'float64')

  c1 = scipy.signal.convolve2d(im1.astype('float64'), Kx, 'same', 'symm')
  c2 = scipy.signal.convolve2d(im2.astype('float64'), Kx, 'same', 'symm')
  c3 = scipy.signal.convolve2d(im3.astype('float64'), Kx, 'same', 'symm')

  return c1 + 2*c2 + c3

def Central_Ey(im1, im2, im3):
  """Calculates the approximate central derivative in Y direction"""
  
  Ky = numpy.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], 'float64')

  c1 = numpy.ndarray(im1.shape, 'float64')
  c2 = numpy.ndarray(im2.shape, 'float64')
  c3 = numpy.ndarray(im3.shape, 'float64')

  c1 = scipy.signal.convolve2d(im1.astype('float64'), Ky, 'same', 'symm')
  c2 = scipy.signal.convolve2d(im2.astype('float64'), Ky, 'same', 'symm')
  c3 = scipy.signal.convolve2d(im3.astype('float64'), Ky, 'same', 'symm')

  return c1 + 2*c2 + c3

def Central_Et(im1, im2, im3):
  """Calculates the approximate central derivative in Y direction"""
  
  Kt = numpy.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 'float64')

  c1 = numpy.zeros(im1.shape, 'float64')
  c3 = numpy.zeros(im3.shape, 'float64')

  c1 = scipy.signal.convolve2d(im1.astype('float64'), Kt, 'same', 'symm')
  c3 = scipy.signal.convolve2d(im3.astype('float64'), Kt, 'same', 'symm')

  return c3 - c1

class GradientTest(unittest.TestCase):
  """Performs various combined temporal gradient tests."""

  def test01_HornAndSchunckCxxAgainstPythonSynthetic(self):
    
    i1, i2 = make_image_pair_1()
    grad = bob.ip.HornAndSchunckGradient(i1.shape)
    ex_cxx, ey_cxx, et_cxx = grad(i1, i2)
    ex_python = Forward_Ex(i1, i2)
    ey_python = Forward_Ey(i1, i2)
    et_python = Forward_Et(i1, i2)
    self.assertTrue( numpy.array_equal(ex_cxx, ex_python) )
    self.assertTrue( numpy.array_equal(ey_cxx, ey_python) )
    self.assertTrue( numpy.array_equal(et_cxx, et_python) )

    i1, i2 = make_image_pair_2()
    grad.shape = i1.shape
    ex_cxx, ey_cxx, et_cxx = grad(i1, i2)
    ex_python = Forward_Ex(i1, i2)
    ey_python = Forward_Ey(i1, i2)
    et_python = Forward_Et(i1, i2)
    self.assertTrue( numpy.array_equal(ex_cxx, ex_python) )
    self.assertTrue( numpy.array_equal(ey_cxx, ey_python) )
    self.assertTrue( numpy.array_equal(et_cxx, et_python) )

  def test02_SobelCxxAgainstPythonSynthetic(self):
    
    i1, i2, i3 = make_image_tripplet_1()
    grad = bob.ip.SobelGradient(i1.shape)
    ex_python = Central_Ex(i1, i2, i3)
    ey_python = Central_Ey(i1, i2, i3)
    et_python = Central_Et(i1, i2, i3)
    ex_cxx, ey_cxx, et_cxx = grad(i1, i2, i3)
    self.assertTrue( numpy.array_equal(ex_cxx, ex_python) )
    self.assertTrue( numpy.array_equal(ey_cxx, ey_python) )
    self.assertTrue( numpy.array_equal(et_cxx, et_python) )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(GradientTest)
