#!/usr/bin/env python
#
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# 24 Nov 2010

import os, sys
import unittest
import torch
import random
import math

#############################################################################
# Compare naive DCT/DFT implementation with fast FCT/FFT implementation
# based on oourafft
#############################################################################

def compare(v1, v2, width):
  return abs(v1-v2) <= width


def test_fct1D(N, t, eps, obj):
  # process using DCT
  dct = torch.sp.spDCT()
  dct.process(t)
  obj.assertEqual(dct.getNOutputs(), 1)

  # process using FCT
  fct = torch.sp.spFCT_oourafft()
  fct.process(t)
  obj.assertEqual(fct.getNOutputs(), 1)

  # get answers and compare them
  dt_dct = dct.getOutput(0)
  dt_fct = fct.getOutput(0)
  for i in range(N):
    obj.assertTrue(compare(dt_dct.get(i), dt_fct.get(i), 1e-3))

  # process using inverse FCT
  ifct = torch.sp.spFCT_oourafft(True)
  ifct.process(dt_fct)
  obj.assertEqual(ifct.getNOutputs(), 1)

  # get answer and compare to original
  idt_fct = ifct.getOutput(0)
  for i in range(N):
    obj.assertTrue(compare(idt_fct.get(i), t.get(i), 1e-3))


def test_fct2D(M, N, t, eps, obj):
  # process using DCT
  dct = torch.sp.spDCT()
  dct.process(t)
  obj.assertEqual(dct.getNOutputs(), 1)

  # process using FCT
  fct = torch.sp.spFCT_oourafft()
  fct.process(t)
  obj.assertEqual(fct.getNOutputs(), 1)

  # get answers and compare them
  dt_dct = dct.getOutput(0)
  dt_fct = fct.getOutput(0)
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(dt_dct.get(i,j), dt_fct.get(i,j), 1e-3))

  # process using inverse FCT
  ifct = torch.sp.spFCT_oourafft(True)
  ifct.process(dt_fct)
  obj.assertEqual(ifct.getNOutputs(), 1)

  # get answer and compare to original
  idt_fct = ifct.getOutput(0)
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(idt_fct.get(i,j), t.get(i,j), 1e-3))


def test_fft1D(N, t, eps, obj):
  # process using DFT
  dft = torch.sp.spDFT()
  dft.process(t)
  obj.assertEqual(dft.getNOutputs(), 1)

  # process using FFT
  fft = torch.sp.spFFT_oourafft()
  fft.process(t)
  obj.assertEqual(fft.getNOutputs(), 1)

  # get answers and compare them
  dt_dft = dft.getOutput(0)
  dt_fft = fft.getOutput(0)
  for i in range(N):
    for j in range(2):
      obj.assertTrue(compare(dt_dft.get(i,j), dt_fft.get(i,j), 1e-3))

  # process using inverse FFT
  ifft = torch.sp.spFFT_oourafft(True)
  ifft.process(dt_fft)
  obj.assertEqual(ifft.getNOutputs(), 1)

  # get answer and compare to original
  idt_fft = ifft.getOutput(0)
  for i in range(N):
    obj.assertTrue(compare(idt_fft.get(i), t.get(i), 1e-3))


def test_fft2D(M, N, t, eps, obj):
  # process using DFT
  dft = torch.sp.spDFT()
  dft.process(t)
  obj.assertEqual(dft.getNOutputs(), 1)

  # process using FFT
  fft = torch.sp.spFFT_oourafft()
  fft.process(t)
  obj.assertEqual(fft.getNOutputs(), 1)

  # get answers and compare them
  dt_dft = dft.getOutput(0)
  dt_fft = fft.getOutput(0)
  for i in range(M):
    for j in range(N):
      for k in range(2):
        obj.assertTrue(compare(dt_dft.get(i,j,k), dt_fft.get(i,j,k), 1e-3))

  # process using inverse FFT
  ifft = torch.sp.spFFT_oourafft(True)
  ifft.process(dt_fft)
  obj.assertEqual(ifft.getNOutputs(), 1)

  # get answer and compare to original
  idt_fft = ifft.getOutput(0)
  for i in range(M):
    for j in range(N):
      obj.assertTrue(compare(idt_fft.get(i,j), t.get(i,j), 1e-3))



##################### Unit Tests ##################  
class TransformTest(unittest.TestCase):
  """Performs for dct, dct2, fft, fft2 and their inverses"""

##################### DCT Tests ##################  
  def test_fct1D_2to64_set(self):
    # size of the data
    for Nexp in range(1,7):
      N = 2**Nexp
      # set up simple 1D tensor
      t = torch.core.FloatTensor(N)
      for i in range(N):
        t.set(i, 1.0+i)

      # call the test function
      test_fct1D(N, t, 1e-3, self)

  def test_fct1D_range2to2048_random(self):
    # This tests the 1D FCT using 10 random vectors
    # The size of each vector is randomly chosen between 3 and 2048
    for loop in range(0,10):
      # size of the data
      Nexp = random.randint(1,12)
      N = 2**Nexp

      # set up simple 1D random tensor 
      t = torch.core.FloatTensor(N)
      for i in range(N):
        t.set(i,random.uniform(1, 10))

      # call the test function
      test_fct1D(N, t, 1e-3, self)


  def test_fct2D_2x2to8x8_set(self):
    # size of the data
    for Mexp in range(1,4):
      M = 2**Mexp
      for Nexp in range(1,4):
        N = 2**Nexp
        # set up simple 2D tensor
        t = torch.core.FloatTensor(M,N)
        for i in range(M):
          for j in range(N):
            t.set(i, j, 1.+i+j)

        # call the test function
        test_fct2D(M, N, t, 1e-3, self)


  def test_fct2D_range2x2to64x64_random(self):
    # This tests the 2D FCT using 5 random vectors
    # The size of each vector is randomly chosen between 2x2 and 64x64
    for loop in range(0,10):
      # size of the data
      Mexp = random.randint(1,7)
      M = 2**Mexp
      Nexp = random.randint(1,7)
      N = 2**Nexp

      # set up simple 1D random tensor 
      t = torch.core.FloatTensor(M,N)
      for i in range(M):
        for j in range(N):
          t.set(i,j,random.uniform(1, 10))

      # call the test function
      test_fct2D(M, N, t, 1e-3, self)


##################### DFT Tests ##################  
  def test_fft1D_2to64_set(self):
    # size of the data
    for Nexp in range(1,7):
      N = 2**Nexp
      # set up simple 1D tensor
      t = torch.core.FloatTensor(N)
      for i in range(N):
        t.set(i, 1.0+i)

      # call the test function
      test_fft1D(N, t, 1e-3, self)

  def test_fft1D_range2to2048_random(self):
    # This tests the 1D FFT using 10 random vectors
    # The size of each vector is randomly chosen between 3 and 2048
    for loop in range(1,10):
      # size of the data
      N = random.randint(2,2048)
      Nexp = random.randint(0,12)
      N = 2**Nexp

      # set up simple 1D random tensor 
      t = torch.core.FloatTensor(N)
      for i in range(N):
        t.set(i,random.uniform(1, 10))

      # call the test function
      test_fft1D(N, t, 1e-3, self)


  def test_fft2D_2x2to8x8_set(self):
    # size of the data
    for Mexp in range(1,4):
      M = 2**Mexp
      for Nexp in range(1,4):
        N = 2**Nexp
        # set up simple 2D tensor
        t = torch.core.FloatTensor(M,N)
        for i in range(M):
          for j in range(N):
            t.set(i, j, 1.+i+j)

        # call the test function
        test_fft2D(M, N, t, 1e-3, self)


  def test_fft2D_range2x2to64x64_random(self):
    # This tests the 2D FFT using 10 random vectors
    # The size of each vector is randomly chosen between 2x2 and 64x64
    for loop in range(0,10):
      # size of the data
      Mexp = random.randint(1,7)
      M = 2**Mexp
      Nexp = random.randint(1,7)
      N = 2**Nexp

      # set up simple 2D random tensor 
      t = torch.core.FloatTensor(M,N)
      for i in range(M):
        for j in range(N):
          t.set(i,j,random.uniform(1, 10))

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

