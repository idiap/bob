#!/usr/bin/env python
#
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# 24 Nov 2010

import os, sys
import unittest
import torch

#############################################################################
# Compare naive DCT/DFT implementation with values returned by Matlab
#############################################################################

def compare(v1, v2, width):
  return abs(v1-v2) <= width


def test_dct1D(N, t, mat, eps, obj):
  # process using DCT
  dct = torch.sp.spDCT()
  dct.process(t)
  obj.assertEqual(dct.getNOutputs(), 1)

  # get answer and compare to matlabs dct
  dt = dct.getOutput(0)

  for i in range(N):
    obj.assertTrue(compare(dt.get(i), mat[i], 1e-3))

  # process using inverse DCT
  idct = torch.sp.spDCT(True)
  idct.process(dt)
  obj.assertEqual(idct.getNOutputs(), 1)

  # get answer and compare to original
  idt = idct.getOutput(0)
  for i in range(N):
    obj.assertTrue(compare(idt.get(i), t.get(i), 1e-3))


def test_dct2D(N, t, mat, eps, obj):
    # process using DCT
    dct = torch.sp.spDCT()
    dct.process(t)
    obj.assertEqual(dct.getNOutputs(), 1)

    # get answer and compare to matlabs dct2 (warning do not use dct)
    dt = dct.getOutput(0)

    for i in range(N):
      for j in range(N):
        obj.assertTrue(compare(dt.get(i,j), mat[i][j], 1e-3))

    # process using inverse DCT
    idct = torch.sp.spDCT(True)
    idct.process(dt)
    obj.assertEqual(idct.getNOutputs(), 1)

    # get answer and compare to original
    idt = idct.getOutput(0)
    for i in range(N):
      for j in range(N):
        obj.assertTrue(compare(idt.get(i,j), t.get(i,j), 1e-3))


def test_dft1D(N, t, mat, eps, obj):
    # process using DFT
    fft = torch.sp.spDFT()
    fft.process(t)
    obj.assertEqual(fft.getNOutputs(), 1)

    # get answer and compare to matlabs fft
    dt = fft.getOutput(0)

    for i in range(N):
      for j in range(2):
        obj.assertTrue(compare(dt.get(i,j), mat[i][j], 1e-3))

    # process using inverse FFT
    ifft = torch.sp.spDFT(True)
    ifft.process(dt)
    obj.assertEqual(ifft.getNOutputs(), 1)

    # get answer and compare to original
    idt = ifft.getOutput(0)
    for i in range(N):
      obj.assertTrue(compare(idt.get(i), t.get(i), 1e-3))


def test_dft2D(N, t, mat, eps, obj):
    # process using DFT
    fft = torch.sp.spDFT()
    fft.process(t)
    obj.assertEqual(fft.getNOutputs(), 1)

    # get answer and compare to matlabs fft
    dt = fft.getOutput(0)

    for i in range(N):
      for j in range(N):
        for k in range(2):
          obj.assertTrue(compare(dt.get(i,j,k), mat[i][j][k], 1e-3))

    # process using inverse FFT
    ifft = torch.sp.spDFT(True)
    ifft.process(dt)
    obj.assertEqual(ifft.getNOutputs(), 1)

    # get answer and compare to original
    idt = ifft.getOutput(0)
    for i in range(N):
      for j in range(N):
        obj.assertTrue(compare(idt.get(i,j), t.get(i,j), 1e-3))



##################### Unit Tests ##################  
class TransformTest(unittest.TestCase):
  """Performs for dct, dct2, dft, dft2 and their inverses"""

##################### DCT Tests ##################  
  def test_dct1D_3(self):
    # size of the data
    N = 3

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)

    # array containing matlab values
    mat = (3.4641,-1.4142,0.)

    # call the test function
    test_dct1D(N, t, mat, 1e-3, self)


  def test_dct1D_5(self):
    # size of the data
    N = 5

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)

    # array containing matlab values
    mat = (6.7082,-3.1495,0.,-0.2840,0.)

    # call the test function
    test_dct1D(N, t, mat, 1e-3, self)


  def test_dct1D_8(self):
    # size of the data
    N = 8

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)

    # array containing matlab values
    mat = (12.7279,-6.4423,0.,-0.6735,0.,-0.2009,0.,-0.0507)

    # call the test function
    test_dct1D(N, t, mat, 1e-3, self)
  
  
  def test_dct1D_17(self):
    # size of the data
    N = 17

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)

    # array containing matlab values
    mat = (37.1080,-20.0585,0.,-2.2025,0.,-0.7727,0.,-0.3768,
             0.,-0.2116,0.,-0.1249,0.,-0.0713,0.,-0.0326,0.)

    # call the test function
    test_dct1D(N, t, mat, 1e-3, self)
    

  def test_dct2D_2x2a(self):
    # size of the data
    N = 2 

    # set up simple 2D tensor
    t = torch.core.FloatTensor(N, N)
    t.set(0, 0, 1.0)
    t.set(0, 1, 0.0)
    t.set(1, 0, 0.0)
    t.set(1, 1, 0.0)

    # array containing matlab values
    mat = ((0.5, 0.5), (0.5, 0.5))

    # call the test function
    test_dct2D(N, t, mat, 1e-3, self)


  def test_dct2D_2x2b(self):
    # size of the data
    N = 2 

    # set up simple 2D tensor
    t = torch.core.FloatTensor(N, N)
    t.set(0, 0, 3.2)
    t.set(0, 1, 4.7)
    t.set(1, 0, 5.4)
    t.set(1, 1, 0.2)

    # array containing matlab values
    mat = ((6.75, 1.85), (1.15,-3.35))

    # call the test function
    test_dct2D(N, t, mat, 1e-3, self)


  def test_dct2D_4x4(self):
    # size of the data
    N = 4 

    # set up simple tensor
    t = torch.core.FloatTensor(N, N)

    for i in range(N):
      for j in range(N):
        t.set(i, j, 1.0+i+j)

    # array containing matlab values
    mat = ((16.0000, -4.4609, 0., -0.3170), (-4.4609, 0., 0., 0.),
           (0., 0., 0., 0.), (-0.3170, 0., 0., 0.))

    # call the test function
    test_dct2D(N, t, mat, 1e-3, self)


  def test_dct2D_8x8(self):
    # size of the data
    N = 8

    # set up simple tensor (have to be 2d)
    t = torch.core.FloatTensor(N, N)

    for i in range(N):
      for j in range(N):
        t.set(i, j, 1.0+i+j)

    # array containing matlab values
    mat = ((64.0000, -18.2216, 0., -1.9048, 0., -0.5682, 0., -0.1434),
           (-18.2216, 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0.),
           (-1.9048, 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0.),
           (-0.5682, 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0.),
           (-0.1434, 0., 0., 0., 0., 0., 0., 0.))

    # call the test function
    test_dct2D(N, t, mat, 1e-3, self)


  def test_dct2D_16x16(self):
    # size of the data
    N = 16

    # set up simple tensor (have to be 2d)
    t = torch.core.FloatTensor(N, N)

    for i in range(N):
      for j in range(N):
        t.set(i, j, 1.0+i+j)

    # array containing matlab values
    mat = ((256.0000, -73.2461, 0., -8.0301, 0., -2.8063, 0., -1.3582, 
              0., -0.7507, 0., -0.4286, 0., -0.2242, 0., -0.0700),
           (-73.2461, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (-8.0301, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (-2.8063, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (-1.3582, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (-0.7507, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (-0.4286, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (-0.2242, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.),
           (-0.0700, 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.))

    # call the test function
    test_dct2D(N, t, mat, 1e-3, self)


##################### FFT Tests ##################  
  def test_fft1D_2(self):
    # size of the data
    N = 2

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)
    
    # array containing matlab values
    mat = ((3., 0.), (-1., 0.))

    # call the test function
    test_dft1D(N, t, mat, 1e-3, self)


  def test_fft1D_3(self):
    # size of the data
    N = 3

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)
    
    # array containing matlab values
    mat = ((6., 0.), (-1.5, 0.8660), (-1.5, -0.8660))

    # call the test function
    test_dft1D(N, t, mat, 1e-3, self)


  def test_fft1D_4(self):
    # size of the data
    N = 4

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)
    
    # array containing matlab values
    mat = ((10., 0.), (-2., 2.), (-2., 0.), (-2., -2.))

    # call the test function
    test_dft1D(N, t, mat, 1e-3, self)


  def test_fft1D_8(self):
    # size of the data
    N = 8

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)
    
    # array containing matlab values
    mat = ((36.0000, 0.), (-4., 9.6569), (-4., 4.), (-4., 1.6569),
           (-4., 0.), (-4.,-1.6569), (-4., -4.), (-4.,-9.6569))

    # call the test function
    test_dft1D(N, t, mat, 1e-3, self)


  def test_fft1D_16(self):
    # size of the data
    N = 16

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)
    
    # array containing matlab values
    mat = ((136.00, 0.), (-8., 40.2187), (-8., 19.3137), (-8., 11.9728), 
           (-8., 8.), (-8., 5.3454), (-8., 3.3137), (-8., 1.5913),
           (-8., 0.), (-8., -1.5913), (-8., -3.3137), (-8., -5.3454),
           (-8., -8.), (-8., -11.9728), (-8., -19.3137), (-8., -40.2187))

    # call the test function
    test_dft1D(N, t, mat, 1e-3, self)


  def test_fft1D_17(self):
    # size of the data
    N = 17

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N)
    for i in range(N):
      t.set(i, 1.0+i)

    # array containing matlab values
    mat = ((153.,0.), (-8.5,45.4710), (-8.5,21.9410), (-8.5,13.7280),
           (-8.5,9.3241), (-8.5,6.4189), (-8.5,4.2325), (-8.5,2.4185),
           (-8.5,0.7876), (-8.5,-0.7876), (-8.5,-2.4185), (-8.5,-4.2325),
           (-8.5,-6.4189), (-8.5,-9.3241), (-8.5,-13.7280), (-8.5,-21.9410),
           (-8.5,-45.4710))

    # call the test function
    test_dft1D(N, t, mat, 1e-3, self)
    

  def test_fft2D_2x2a(self):
    # size of the data
    N = 2

    # set up simple 1D tensor
    t = torch.core.FloatTensor(N,N)
    for i in range(N):
      for j in range(N):
        t.set(i, j, 1.0+i+j)
    
    # array containing matlab values
    mat = ( ((8., 0.), (-2., 0.)), ((-2., 0.), (0., 0.)) )
  
    # call the test function
    test_dft2D(N, t, mat, 1e-3, self)


  def test_fft2D_2x2b(self):
    # size of the data
    N = 2

    # set up simple 2D tensor
    t = torch.core.FloatTensor(N, N)
    t.set(0, 0, 3.2)
    t.set(0, 1, 4.7)
    t.set(1, 0, 5.4)
    t.set(1, 1, 0.2)

    # array containing matlab values
    mat = ( ((13.5, 0.), (3.7, 0.)), ((2.3, 0.), (-6.7, 0.)) )
  
    # call the test function
    test_dft2D(N, t, mat, 1e-3, self)


  def test_fft2D_3x3(self):
    # size of the data
    M = 3
    N = 3

    # set up simple 2D tensor
    t = torch.core.FloatTensor(M, N)
    for i in range(M):
      for j in range(N):
        t.set(i, j, 1.0+i+j)

    # array containing matlab values
    mat = ( ((27., 0.), (-4.5, 2.5981), (-4.5, -2.5981)),
            ((-4.5, 2.5981), (0., 0.), (0., 0.)),
            ((-4.5000, -2.5981), (0., 0.), (0., 0.)))
  
    # call the test function
    test_dft2D(N, t, mat, 1e-3, self)


  def test_fft2D_4x4(self):
    # size of the data
    N = 4

    # set up simple 2D tensor
    t = torch.core.FloatTensor(N, N)
    for i in range(N):
      for j in range(N):
        t.set(i, j, 1.0+i+j)

    # array containing matlab values
    mat = ( ((64., 0.), (-8., 8.), (-8.,0.), (-8., -8.)), 
            ((-8., 8.), (0., 0.), (0., 0.), (0., 0.)), 
            ((-8., 0.), (0., 0.), (0., 0.), (0., 0.)), 
            ((-8., -8.), (0., 0.), (0., 0.), (0., 0.)))
  
    # call the test function
    test_dft2D(N, t, mat, 1e-3, self)


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

