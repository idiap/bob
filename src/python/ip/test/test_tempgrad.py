#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue  6 Sep 18:13:36 2011 


"""Tests our Temporal Gradient utilities going through some example data.
"""

import os, sys
import unittest
import torch
import math

def load_gray(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join("data", "flow", relative_filename)
  array = torch.io.Array(filename)
  return array.get()[0,:,:] 

def load_known_flow(relative_filename):
  filename = os.path.join("data", "flow", relative_filename)
  array = torch.io.Array(filename)
  data = array.get()
  return data[:,:,0].cast('float64'), data[:,:,1].cast('float64')

def make_image_pair_1():
  """Creates two images for you to calculate the flow
  
  1 1 => 1 1
  1 1    1 2
  
  """
  im1 = torch.core.array.uint8_2(2,2)
  im1.fill(1)
  im2 = torch.core.array.uint8_2(2,2)
  im2.fill(1)
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
  im1 = torch.core.array.uint8_2(5,5)
  im1.fill(10)
  im1[1:3, 1:] = 5
  im2 = torch.core.array.uint8_2(5,5)
  im2.fill(10)
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
  im1 = torch.core.array.uint8_2(5,5)
  im1.fill(10)
  im1[1:3, 1:] = 5
  im2 = torch.core.array.uint8_2(5,5)
  im2.fill(10)
  im2[2:4, 2:] = 5
  im3 = torch.core.array.uint8_2(5,5)
  im3.fill(10)
  im3[3:, 3:] = 5
  return im1, im2, im3

def Forward_U(im1, im2):
  """Calculates the approximate forward derivative in X direction"""
  e = torch.core.array.float64_2(im1.shape())
  e.fill(0) #only last column should keep this value
  for i in range(im1.extent(0)-1):
    for j in range(im1.extent(1)-1):
      e[i,j] = ( im1[i,j+1] - im1[i,j] +
                 im1[i+1,j+1] - im1[i+1,j] +
                 im2[i,j+1] - im2[i,j] +
                 im2[i+1,j+1] - im2[i+1,j] ) / (2*math.sqrt(2))
  for j in range(im1.extent(1)-1): #last row there is no i+1
    e[-1,j] = ( im1[-1,j+1]-im1[-1,j]+im2[-1,j+1]-im2[-1,j] ) / math.sqrt(2)
  return e

def Forward_V(im1, im2):
  """Calculates the approximate forward derivative in Y direction"""
  e = torch.core.array.float64_2(im1.shape())
  e.fill(0) #only last row should keep this value
  for i in range(im1.extent(0)-1):
    for j in range(im1.extent(1)-1):
      e[i,j] = ( im1[i+1,j] - im1[i,j] +
                 im1[i+1,j+1] - im1[i,j+1] +
                 im2[i+1,j] - im2[i,j] +
                 im2[i+1,j+1] - im2[i,j+1] ) / (2*math.sqrt(2))
  for i in range(im1.extent(0)-1): #last column there is no j+1
    e[i,-1] = ( im1[i+1,-1]-im1[i,-1]+im2[i+1,-1]-im2[i,-1] ) / math.sqrt(2)
  return e

def Central_U(im1, im2, im3):
  """Calculates the approximate central derivative in X direction"""

  Kx = torch.core.array.float64_2([-1, 0, +1, -2, 0, +2, -1, 0, +1], (3,3))

  c1 = torch.core.array.float64_2(im1.shape())
  c2 = torch.core.array.float64_2(im2.shape())
  c3 = torch.core.array.float64_2(im3.shape())

  torch.sp.convolve(im1.cast('float64'), Kx, c1, 
      torch.sp.ConvolutionSize.Same, torch.sp.ConvolutionBorder.Mirror)
  torch.sp.convolve(im2.cast('float64'), Kx, c2,
      torch.sp.ConvolutionSize.Same, torch.sp.ConvolutionBorder.Mirror)
  torch.sp.convolve(im3.cast('float64'), Kx, c3,
      torch.sp.ConvolutionSize.Same, torch.sp.ConvolutionBorder.Mirror)

  return c1 + 2*c2 + c3

def Central_V(im1, im2, im3):
  """Calculates the approximate central derivative in Y direction"""
  
  Ky = torch.core.array.float64_2([-1, -2, -1, 0, 0, 0, +1, +2, +1], (3,3))

  c1 = torch.core.array.float64_2(im1.shape())
  c2 = torch.core.array.float64_2(im2.shape())
  c3 = torch.core.array.float64_2(im3.shape())

  torch.sp.convolve(im1.cast('float64'), Ky, c1,
      torch.sp.ConvolutionSize.Same, torch.sp.ConvolutionBorder.Mirror)
  torch.sp.convolve(im2.cast('float64'), Ky, c2,
      torch.sp.ConvolutionSize.Same, torch.sp.ConvolutionBorder.Mirror)
  torch.sp.convolve(im3.cast('float64'), Ky, c3,
      torch.sp.ConvolutionSize.Same, torch.sp.ConvolutionBorder.Mirror)

  return c1 + 2*c2 + c3

class GradientTest(unittest.TestCase):
  """Performs various combined temporal gradient tests."""

  def test01_ForwardCxxAgainstSyntheticPython(self):
    
    i1, i2 = make_image_pair_1()
    grad = torch.ip.ForwardGradient(i1.shape())
    u_cxx, v_cxx = grad(i1, i2)
    u_python = Forward_U(i1, i2)
    v_python = Forward_V(i1, i2)
    self.assertEqual( u_cxx, u_python )
    self.assertEqual( v_cxx, v_python )

    i1, i2 = make_image_pair_2()
    grad = torch.ip.ForwardGradient(i1.shape())
    u_cxx, v_cxx = grad(i1, i2)
    u_python = Forward_U(i1, i2)
    v_python = Forward_V(i1, i2)
    self.assertEqual( u_cxx, u_python )
    self.assertEqual( v_cxx, v_python )

  def test02_CentralCxxAgainstSyntheticPython(self):
    
    i1, i2, i3 = make_image_tripplet_1()
    grad = torch.ip.CentralGradient(i1.shape())
    u_python = Central_U(i1, i2, i3)
    v_python = Central_V(i1, i2, i3)
    u_cxx, v_cxx = grad(i1, i2, i3)
    self.assertEqual( u_cxx, u_python )
    self.assertEqual( v_cxx, v_python )

  def notest03_FlowTest(self):

    if1 = os.path.join("rubberwhale", "frame10_gray.png")
    if2 = os.path.join("rubberwhale", "frame11_gray.png")
    i1 = load_gray(if1)
    i2 = load_gray(if2)

    grad = torch.ip.ForwardGradient(i1.shape())
    (u, v) = grad(i1, i2)
    array = torch.ip.flowutils.flow2hsv(u,v).convert('uint8', sourceRange=(0.,1.))
    array.save("rubberwhale-forwardgrad.png")

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

