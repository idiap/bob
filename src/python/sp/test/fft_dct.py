#!/usr/bin/env python
#
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# 24 Nov 2010

import os, sys
import unittest
import torch
import random

#############################################################################
# Test fast DCT/FFT implementation based on FFTPACK
#############################################################################

def compare(v1, v2, width):
  return abs(v1-v2) <= width


def test_dct1D(N, t, eps, obj):
  # process using DCT
  u_dct = torch.core.array.float64_1(N)
  dct = torch.sp.DCT1D(N)
  dct(t,u_dct)

  # process using inverse DCT 
  u_dct_idct = torch.core.array.float64_1(N)
  idct = torch.sp.IDCT1D(N)
  idct(u_dct,u_dct_idct)

  # get answer and compare to original
  for i in range(N):
    obj.assertTrue(compare(u_dct_idct[i], t[i], 1e-3))


def test_dct2D(M, N, t, eps, obj):
  # process using DCT
  u_dct = torch.core.array.float64_2(M,N)
  dct = torch.sp.DCT2D(M,N)
  dct(t,u_dct)

  # process using inverse DCT 
  u_dct_idct = torch.core.array.float64_2(M,N)
  idct = torch.sp.IDCT2D(M,N)
  idct(u_dct,u_dct_idct)

  # get answer and compare to original
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(u_dct_idct[i,j], t[i,j], 1e-3))


def test_fft1D(N, t, eps, obj):
  # process using FFT
  u_fft = torch.core.array.complex128_1(N)
  fft = torch.sp.FFT1D(N)
  fft(t,u_fft)

  # process using inverse FFT 
  u_fft_ifft = torch.core.array.complex128_1(N)
  ifft = torch.sp.IFFT1D(N)
  ifft(u_fft,u_fft_ifft)

  # get answer and compare to original
  for i in range(N):
    obj.assertTrue(compare(u_fft_ifft[i], t[i], 1e-3))


def test_fft2D(M, N, t, eps, obj):
  # process using FFT
  u_fft = torch.core.array.complex128_2(M,N)
  fft = torch.sp.FFT2D(M,N)
  fft(t,u_fft)

  # process using inverse FFT 
  u_fft_ifft = torch.core.array.complex128_2(M,N)
  ifft = torch.sp.IFFT2D(M,N)
  ifft(u_fft,u_fft_ifft)

  # get answer and compare to original
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(u_fft_ifft[i,j], t[i,j], 1e-3))



##################### Unit Tests ##################  
class TransformTest(unittest.TestCase):
  """Performs for dct, dct2, fft, fft2 and their inverses"""

##################### DCT Tests ##################  
  def test_dct1D_1to64_set(self):
    # size of the data
    for N in range(1,65):
      # set up simple 1D tensor
      t = torch.core.array.float64_1(N)
      for i in range(N):
        t[i] = 1.0+i

      # call the test function
      test_dct1D(N, t, 1e-3, self)

  def test_dct1D_range1to2048_random(self):
    # This tests the 1D FCT using 10 random vectors
    # The size of each vector is randomly chosen between 3 and 2048
    for loop in range(0,10):
      # size of the data
      N = random.randint(1,2048)

      # set up simple 1D random tensor 
      t = torch.core.array.float64_1(N)
      for i in range(N):
        t[i] = random.uniform(1, 10)

      # call the test function
      test_dct1D(N, t, 1e-3, self)


  def test_dct2D_1x1to8x8_set(self):
    # size of the data
    for M in range(1,9):
      for N in range(1,9):
        # set up simple 2D tensor
        t = torch.core.array.float64_2(M,N)
        for i in range(M):
          for j in range(N):
            t[i,j] = 1.+i+j

        # call the test function
        test_dct2D(M, N, t, 1e-3, self)


  def test_dct2D_range1x1to64x64_random(self):
    # This tests the 2D FCT using 5 random vectors
    # The size of each vector is randomly chosen between 2x2 and 64x64
    for loop in range(0,10):
      # size of the data
      M = random.randint(1,64)
      N = random.randint(1,64)

      # set up simple 1D random tensor 
      t = torch.core.array.float64_2(M,N)
      for i in range(M):
        for j in range(N):
          t[i,j] = random.uniform(1, 10)

      # call the test function
      test_dct2D(M, N, t, 1e-3, self)


##################### DFT Tests ##################  
  def test_fft1D_1to64_set(self):
    # size of the data
    for N in range(1,65):
      # set up simple 1D tensor
      t = torch.core.array.complex128_1(N)
      for i in range(N):
        t[i] = 1.0+i

      # call the test function
      test_fft1D(N, t, 1e-3, self)

  def test_fft1D_range1to2048_random(self):
    # This tests the 1D FFT using 10 random vectors
    # The size of each vector is randomly chosen between 3 and 2048
    for loop in range(0,10):
      # size of the data
      N = random.randint(1,2048)

      # set up simple 1D random tensor 
      t = torch.core.array.complex128_1(N)
      for i in range(N):
        t[i] = random.uniform(1, 10)

      # call the test function
      test_fft1D(N, t, 1e-3, self)


  def test_fft2D_1x1to8x8_set(self):
    # size of the data
    for M in range(1,9):
      for N in range(1,9):
        # set up simple 2D tensor
        t = torch.core.array.complex128_2(M,N)
        for i in range(M):
          for j in range(N):
            t[i,j] = 1.+i+j

        # call the test function
        test_fft2D(M, N, t, 1e-3, self)


  def test_fft2D_range1x1to64x64_random(self):
    # This tests the 2D FFT using 10 random vectors
    # The size of each vector is randomly chosen between 2x2 and 64x64
    for loop in range(0,10):
      # size of the data
      M = random.randint(1,64)
      N = random.randint(1,64)

      # set up simple 2D random tensor 
      t = torch.core.array.complex128_2(M,N)
      for i in range(M):
        for j in range(N):
          t[i,j] = random.uniform(1, 10)

      # call the test function
      test_fft2D(M, N, t, 1e-3, self)


##################### Main ##################  
if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

