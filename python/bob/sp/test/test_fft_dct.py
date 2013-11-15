#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Nov 14 23:47:36 CET 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

import os, sys
import unittest
import numpy
import random
from .. import *

#############################################################################
# Test fast DCT/FFT implementation based on numpy FFT
#############################################################################

def compare(v1, v2, width):
  return abs(v1-v2) <= width


def _dct1D(N, t, eps, obj):
  # process using DCT
  u_dct = numpy.zeros((N,), 'float64')
  dct = DCT1D(N)
  dct(t,u_dct)

  # process using inverse DCT 
  u_dct_idct = numpy.zeros((N,), 'float64')
  idct = IDCT1D(N)
  idct(u_dct,u_dct_idct)

  # get answer and compare to original
  for i in range(N):
    obj.assertTrue(compare(u_dct_idct[i], t[i], 1e-3))

  v_dct_idct = idct(dct(t))
  # get answer and compare to original
  for i in range(N):
    obj.assertTrue(compare(v_dct_idct[i], t[i], 1e-3))


def _dct2D(M, N, t, eps, obj):
  # process using DCT
  u_dct = numpy.zeros((M,N), 'float64')
  dct = DCT2D(M,N)
  dct(t,u_dct)

  # process using inverse DCT 
  u_dct_idct = numpy.zeros((M,N), 'float64')
  idct = IDCT2D(M,N)
  idct(u_dct,u_dct_idct)

  # get answer and compare to original
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(u_dct_idct[i,j], t[i,j], 1e-3))

  v_dct_idct = idct(dct(t))
  # get answer and compare to original
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(u_dct_idct[i,j], t[i,j], 1e-3))



def _fft1D(N, t, eps, obj):
  # process using FFT
  u_fft = numpy.zeros((N,), 'complex128')
  fft = FFT1D(N)
  fft(t,u_fft)

  # process using inverse FFT 
  u_fft_ifft = numpy.zeros((N,), 'complex128')
  ifft = IFFT1D(N)
  ifft(u_fft,u_fft_ifft)

  # get answer and compare to original
  for i in range(N):
    obj.assertTrue(compare(u_fft_ifft[i], t[i], 1e-3))

  v_fft_ifft = ifft(fft(t))
  # get answer and compare to original
  for i in range(N):
    obj.assertTrue(compare(v_fft_ifft[i], t[i], 1e-3))


def _fft2D(M, N, t, eps, obj):
  # process using FFT
  u_fft = numpy.zeros((M,N), 'complex128')
  fft = FFT2D(M,N)
  fft(t,u_fft)

  # process using inverse FFT 
  u_fft_ifft = numpy.zeros((M,N), 'complex128')
  ifft = IFFT2D(M,N)
  ifft(u_fft,u_fft_ifft)

  # get answer and compare to original
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(u_fft_ifft[i,j], t[i,j], 1e-3))

  v_fft_ifft = ifft(fft(t))
  # get answer and compare to original
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(v_fft_ifft[i,j], t[i,j], 1e-3))


##################### Unit Tests ##################  
class TransformTest(unittest.TestCase):
  """Performs for dct, dct2, fft, fft2 and their inverses"""

##################### DCT Tests ##################  
  def test_dct1D_1to64_set(self):
    # size of the data
    for N in range(1,65):
      # set up simple 1D tensor
      t = numpy.zeros((N,), 'float64')
      for i in range(N):
        t[i] = 1.0+i

      # call the test function
      _dct1D(N, t, 1e-3, self)

  def test_dct1D_range1to2048_random(self):
    # This tests the 1D FCT using 10 random vectors
    # The size of each vector is randomly chosen between 3 and 2048
    for loop in range(0,10):
      # size of the data
      N = random.randint(1,2048)

      # set up simple 1D random tensor 
      t = numpy.zeros((N,), 'float64')
      for i in range(N):
        t[i] = random.uniform(1, 10)

      # call the test function
      _dct1D(N, t, 1e-3, self)


  def test_dct2D_1x1to8x8_set(self):
    # size of the data
    for M in range(1,9):
      for N in range(1,9):
        # set up simple 2D tensor
        t = numpy.zeros((M,N), 'float64')
        for i in range(M):
          for j in range(N):
            t[i,j] = 1.+i+j

        # call the test function
        _dct2D(M, N, t, 1e-3, self)


  def test_dct2D_range1x1to64x64_random(self):
    # This tests the 2D FCT using 5 random vectors
    # The size of each vector is randomly chosen between 2x2 and 64x64
    for loop in range(0,10):
      # size of the data
      M = random.randint(1,64)
      N = random.randint(1,64)

      # set up simple 1D random tensor 
      t = numpy.zeros((M,N), 'float64')
      for i in range(M):
        for j in range(N):
          t[i,j] = random.uniform(1, 10)

      # call the test function
      _dct2D(M, N, t, 1e-3, self)


##################### DFT Tests ##################  
  def test_fft1D_1to64_set(self):
    # size of the data
    for N in range(1,65):
      # set up simple 1D tensor
      t = numpy.zeros((N,), 'complex128')
      for i in range(N):
        t[i] = 1.0+i

      # call the test function
      _fft1D(N, t, 1e-3, self)

  def test_fft1D_range1to2048_random(self):
    # This tests the 1D FFT using 10 random vectors
    # The size of each vector is randomly chosen between 3 and 2048
    for loop in range(0,10):
      # size of the data
      N = random.randint(1,2048)

      # set up simple 1D random tensor 
      t = numpy.zeros((N,), 'complex128')
      for i in range(N):
        t[i] = random.uniform(1, 10)

      # call the test function
      _fft1D(N, t, 1e-3, self)


  def test_fft2D_1x1to8x8_set(self):
    # size of the data
    for M in range(1,9):
      for N in range(1,9):
        # set up simple 2D tensor
        t = numpy.zeros((M,N), 'complex128')
        for i in range(M):
          for j in range(N):
            t[i,j] = 1.+i+j

        # call the test function
        _fft2D(M, N, t, 1e-3, self)


  def test_fft2D_range1x1to64x64_random(self):
    # This tests the 2D FFT using 10 random vectors
    # The size of each vector is randomly chosen between 2x2 and 64x64
    for loop in range(0,10):
      # size of the data
      M = random.randint(1,64)
      N = random.randint(1,64)

      # set up simple 2D random tensor 
      t = numpy.zeros((M,N), 'complex128')
      for i in range(M):
        for j in range(N):
          t[i,j] = random.uniform(1, 10)

      # call the test function
      _fft2D(M, N, t, 1e-3, self)

  def test_dct1d_methods(self):
    r = numpy.random.randn(7).astype(numpy.float64)
    # 1.a DCT1D
    a = DCT1D(7)
    o_i = a(r)
    b = DCT1D(7)
    c = DCT1D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.length = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,))
    o_f = a(r)
    self.assertTrue( numpy.allclose(o_i, o_f) )
    # 1.b IDCT1D
    a = IDCT1D(7)
    o_i = a(r)
    b = IDCT1D(7)
    c = IDCT1D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.length = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,))
    o_f = a(r)
    self.assertTrue( numpy.allclose(o_i, o_f) )

  def test_dct2d_methods(self):
    r = numpy.random.randn(7,9).astype(numpy.float64)
    # 2.a DCT2D
    a = DCT2D(7,9)
    o_i = a(r)
    b = DCT2D(7,9)
    c = DCT2D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.height = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,9))
    a.width = 10
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b ) 
    o_f = a(r)
    self.assertTrue( numpy.allclose(o_i, o_f) )
    # 2.b IDCT2D
    a = IDCT2D(7,9)
    o_i = a(r)
    b = IDCT2D(7,9)
    c = IDCT2D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.height = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,9))
    a.width = 10
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b ) 
    o_f = a(r)
    self.assertTrue( numpy.allclose(o_i, o_f) )

  def test_fft1d_methods(self):
    v = numpy.random.randn(7).astype(numpy.complex128)
    # 3.a FFT1D
    a = FFT1D(7)
    o_i = a(v)
    b = FFT1D(7)
    c = FFT1D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.length = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,))
    o_f = a(v)
    self.assertTrue( numpy.allclose(o_i, o_f) )
    # 3.b IFFT1D
    a = IFFT1D(7)
    o_i = a(v)
    b = IFFT1D(7)
    c = IFFT1D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.length = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,))
    o_f = a(v)
    self.assertTrue( numpy.allclose(o_i, o_f) )
    
  def test_fft2d_methods(self):
    v = numpy.random.randn(7,9).astype(numpy.complex128)
    # 4.a FFT2D
    a = FFT2D(7,9)
    o_i = a(v)
    b = FFT2D(7,9)
    c = FFT2D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.height = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,9))
    a.width = 10
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b ) 
    o_f = a(v)
    self.assertTrue( numpy.allclose(o_i, o_f) )

    # 4.b IFFT2D
    a = IFFT2D(7,9)
    o_i = a(v)
    b = IFFT2D(7,9)
    c = IFFT2D(a)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertFalse( a != c )
    a.height = 8
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b )
    self.assertTrue(a.shape == (7,9))
    a.width = 10
    self.assertFalse( a == b )
    self.assertTrue( a != b )
    a.shape = (7,9)
    self.assertTrue( a == b )
    self.assertTrue( a == c )
    self.assertFalse( a != b ) 
    o_f = a(v)
    self.assertTrue( numpy.allclose(o_i, o_f) )
