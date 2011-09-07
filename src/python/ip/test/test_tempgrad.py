#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue  6 Sep 18:13:36 2011 


"""Tests our Temporal Gradient utilities going through some example data.
"""

import os, sys
import unittest
import torch

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

def Forward_U(im1, im2):
  """Calculates the approximate derivative in X direction"""
  e = torch.core.array.float64_2(im1.shape())
  e.fill(0) #only last column should keep this value
  for i in range(im1.extent(0)-1):
    for j in range(im1.extent(1)-1):
      e[i,j] = 0.25 * ( im1[i,j+1] - im1[i,j] +
                        im1[i+1,j+1] - im1[i+1,j] +
                        im2[i,j+1] - im2[i,j] +
                        im2[i+1,j+1] - im2[i+1,j] )
  for j in range(im1.extent(1)-1): #last row there is no i+1
    e[-1,j] = 0.5 * ( im1[-1,j+1]-im1[-1,j]+im2[-1,j+1]-im2[-1,j] )
  return e

def Forward_V(im1, im2):
  """Calculates the approximate derivative in Y direction"""
  e = torch.core.array.float64_2(im1.shape())
  e.fill(0) #only last row should keep this value
  for i in range(im1.extent(0)-1):
    for j in range(im1.extent(1)-1):
      e[i,j] = 0.25 * ( im1[i+1,j] - im1[i,j] +
                        im1[i+1,j+1] - im1[i,j+1] +
                        im2[i+1,j] - im2[i,j] +
                        im2[i+1,j+1] - im2[i,j+1] )
  for i in range(im1.extent(0)-1): #last column there is no j+1
    e[i,-1] = 0.5 * ( im1[i+1,-1]-im1[i,-1]+im2[i+1,-1]-im2[i,-1] )
  return e

class GradientTest(unittest.TestCase):
  """Performs various combined temporal gradient tests."""

  def test01_CxxAgainstSyntheticPython(self):
    
    i1, i2 = make_image_pair_1()
    u_cxx, v_cxx = torch.ip.ForwardGradient(i1, i2)

    u_py = torch.core.array.float64_2(i1.shape()); u_py.fill(0)
    v_py = torch.core.array.float64_2(i1.shape()); v_py.fill(0)

    self.assertEqual(u_cxx, u_py)
    self.assertEqual(v_cxx, v_py)

  def notest01_VanillaHornAndSchunckDemo(self):
    
    N = 64
    alpha = 2 

    if1 = os.path.join("rubberwhale", "frame10_gray.png")
    if2 = os.path.join("rubberwhale", "frame11_gray.png")
    i1 = load_gray(if1)
    i2 = load_gray(if2)

    u = torch.core.array.float64_2(i1.shape()); u.fill(0)
    v = torch.core.array.float64_2(i1.shape()); v.fill(0)
    for k in range(N): 
      torch.ip.evalHornAndSchunckFlow(alpha, 1, i1, i2, u, v)
      array = torch.io.Array(torch.ip.flowutils.flow2hsv(u,v))
      array.save("hs_rubberwhale-%d.png" % k)

  def notest02_VanillaHornAndSchunckAgainstOpenCV(self):
    
    # Tests and examplifies usage of the vanilla HS algorithm while comparing
    # the C++ implementation to a OpenCV implementation of the same algorithm.
    # Please note that the comparision is not fair as the OpenCV uses
    # different kernels for the evaluation of the image gradient.

    # This test is only here to help you debug problems.

    # We create a new estimator specifying the alpha parameter (first value)
    # and the number of iterations to perform (second value).
    N = 64
    alpha = 15 

    # The OpticalFlow estimator always receives a blitz::Array<uint8_t,2> as
    # the image input. The output has the same rank and extents but is in
    # doubles.
    
    # This will load the test images: Rubber Whale sequence
    if1 = os.path.join("rubberwhale", "frame10_gray.png")
    if2 = os.path.join("rubberwhale", "frame11_gray.png")
    i1 = load_gray(if1)
    i2 = load_gray(if2)

    u_cxx = torch.core.array.float64_2(i1.shape()); u_cxx.fill(0)
    v_cxx = torch.core.array.float64_2(i1.shape()); v_cxx.fill(0)
    se2 = torch.core.array.float64_2() #squared smoothness error (from cxx)
    be = torch.core.array.float64_2() #brightness error (from cxx)
    torch.ip.evalHornAndSchunckFlow(alpha, N, i1, i2, u_cxx, v_cxx)
    torch.ip.evalHornAndSchunckEc2(u_cxx, v_cxx, se2)
    torch.ip.evalHornAndSchunckFlowEb(i1, i2, u_cxx, v_cxx, be)
    avg_err = (se2 * (alpha**2) + be**2).sum()
    #print "Torch H&S Error (%2d iterations) : %.3e %.3e %.3e" % (N, se2.sum()**0.5, be.sum(), avg_err**0.5)

    u_ocv1, v_ocv1 = compute_flow_opencv(alpha, N/4, if1, if2)
    u_ocv2, v_ocv2 = compute_flow_opencv(alpha, N/2, if1, if2)
    u_ocv, v_ocv = compute_flow_opencv(alpha, N, if1, if2)
    torch.ip.evalHornAndSchunckEc2(u_ocv, v_ocv, se2)
    torch.ip.evalHornAndSchunckEb(i1, i2, u_ocv, v_ocv, be)
    avg_err = (se2 * (alpha**2) + be**2).sum()
    #print "OpenCV H&S Error (%2d iterations): %.3e %.3e %.3e" % (N, se2.sum()**0.5, be.sum(), avg_err**0.5)

    u_c = u_cxx/u_ocv
    v_c = v_cxx/v_ocv
    self.assertTrue(u_c.mean() < 1.1) #check for within 10%
    self.assertTrue(v_c.mean() < 1.1) #check for within 10%
    #print "mean(u_ratio), mean(v_ratio): %.3e %.3e" % (u_c.mean(), v_c.mean()),
    #print "(as close to 1 as possible)"

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

