#!/usr/bin/env python
#
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# 29 Jan 2011

import os, sys
import unittest
import torch
import random

#############################################################################
# Compare naive DCT/DFT implementation with fast FCT/FFT implementation
# based on FFTPACK
#############################################################################

def compare(v1, v2, width):
  return abs(v1-v2) <= width


def test_fft1D(N, t, eps, obj):
  # process using DFT
  # TODO

  # process using FFT
  dt_fft = torch.sp.fft(t)
  obj.assertEqual(dt_fft.dimensions(), 1)

  # get answers and compare them
  # TODO

  # process using inverse FFT
  dt_ifft = torch.sp.ifft(dt_fft)
  obj.assertEqual(dt_ifft.dimensions(), 1)

  # get answer and compare to original
  for i in range(N):
    obj.assertTrue(compare(dt_ifft[i], t[i], 1e-3))


def test_fft2D(M, N, t, eps, obj):
  # process using DFT
  # TODO

  # process using FFT
  dt_fft = torch.sp.fft(t)
  obj.assertEqual(dt_fft.dimensions(), 2)

  # get answers and compare them
  # TODO

  # process using inverse FFT
  dt_ifft = torch.sp.ifft(dt_fft)
  obj.assertEqual(dt_ifft.dimensions(), 2)

  # get answer and compare to original
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(dt_ifft[i,j], t[i,j], 1e-3))



##################### Unit Tests ##################  
class TransformTest(unittest.TestCase):
  """Performs for dct, dct2, fft, fft2 and their inverses"""

##################### DCT Tests ##################  
## TODO:


##################### DFT Tests ##################  
  def test_fft1D_1to64_set(self):
    # size of the data
    for N in range(1,65):
      # set up simple 1D tensor
      t = torch.core.array.complex128_1(N)
      for i in range(N):
        t[i] = complex(1.0+i,0)

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
        t[i] = complex(random.uniform(1, 10),0)

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
            t[i,j] = complex(1.+i+j,0)

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
          t[i,j] = complex(random.uniform(1, 10),0)

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

