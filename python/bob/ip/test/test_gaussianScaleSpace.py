#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Sep 16 16:44:00 2012 +0200
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

"""Tests the Gaussian Scale Space
"""

import os, sys
import unittest
import bob
import numpy
import pkg_resources


eps = 1e-4

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

class GaussianScaleSpaceTest(unittest.TestCase):
  """Performs various tests for the bob.ip.GaussianScaleSpace class"""

  def test01_parametrization(self):
    # Parametrization tests
    op = bob.ip.GaussianScaleSpace(200,250,4,3,-1,0.5,1.6,4.)
    self.assertEqual(op.height, 200)
    self.assertEqual(op.width, 250)
    self.assertEqual(op.n_octaves, 4)
    self.assertEqual(op.n_intervals, 3)
    self.assertEqual(op.octave_min, -1)
    self.assertEqual(op.sigma_n, 0.5)
    self.assertEqual(op.sigma0, 1.6)
    self.assertEqual(op.kernel_radius_factor, 4.)
    op.height = 300
    op.width = 350
    op.n_octaves = 3
    op.n_intervals = 4
    op.octave_min = 0
    op.sigma_n = 0.6
    op.sigma0 = 2.
    op.kernel_radius_factor = 3.
    self.assertEqual(op.height, 300)
    self.assertEqual(op.width, 350)
    self.assertEqual(op.n_octaves, 3)
    self.assertEqual(op.n_intervals, 4)
    self.assertEqual(op.octave_min, 0)
    self.assertEqual(op.sigma_n, 0.6)
    self.assertEqual(op.sigma0, 2.)
    self.assertEqual(op.kernel_radius_factor, 3.)
  
  def test02_processing(self):
    # Processing tests
    A = bob.io.load(F(os.path.join("sift", "vlimg_ref.pgm")))
    No = 3
    Ns = 3
    sigma0 = 1.6
    sigma_n = 0.5
    f=4.
    op = bob.ip.GaussianScaleSpace(A.shape[0],A.shape[1],No,Ns,0,sigma_n,sigma0,f)
    pyr = op(A)

    import math
    # Assumes that octave_min = 0
    dsigma0 = sigma0 * math.sqrt(1.-math.pow(2,-2./float(Ns)))
    Aa = A
    for o in range(No):
      for s in range(-1,Ns+2):
        # Get Gaussian for this scale
        g_pyr = op.get_gaussian(s+1)
        
        # Filtering step
        if s!=-1 or (o==0 and s==-1):
          # Compute scale and radius
          if(o==0 and s==-1):
            sa = sigma0 #* math.pow(2.,s/float(Ns))
            sb = sigma_n
            sigma = math.sqrt(sa*sa - sb*sb)
          else:
            sigma = dsigma0 * math.pow(2,s/float(Ns))
          radius = int(math.ceil(f*sigma))
          # Check values
          self.assertTrue( abs(sigma - g_pyr.sigma_y) < eps)
          self.assertTrue( abs(sigma - g_pyr.sigma_x) < eps)
          self.assertTrue( abs(radius - g_pyr.radius_y) < eps)
          self.assertTrue( abs(radius - g_pyr.radius_x) < eps)
          
          g = bob.ip.Gaussian(radius, radius, sigma, sigma)
          B = g(Aa)
        # Downsampling step
        else:
          # Select image as by VLfeat (seems wrong to me)
          Aa = pyr[o-1][Ns,:,:]
          # Downsample using a trick to make sure that if the length is l=2p+1,
          # the new one is p and not p+1.
          B = Aa[:2*(Aa.shape[0]/2):2,:2*(Aa.shape[1]/2):2]

        # Compare image of the pyramids (Python implementation vs. C++)
        Bpyr = pyr[o][s+1,:,:]
        self.assertEqual(numpy.allclose(B, Bpyr, eps), True)
        Aa = B
        ##For saving/visualizing images
        #base_dir = '/home/user'
        #bob.io.save(Bpyr.astype('uint8'), os.path.join(base_dir, 'pyr_o'+str(o)+'_s'+str(s+1)+'.pgm'))

  def test03_comparison(self):
    # Comparisons tests
    op1 = bob.ip.GaussianScaleSpace(200,250,4,3,-1,0.5,1.6,4.)
    op1b = bob.ip.GaussianScaleSpace(200,250,4,3,-1,0.5,1.6,4.)
    op2 = bob.ip.GaussianScaleSpace(300,250,4,3,-1,0.5,1.6,4.)
    op3 = bob.ip.GaussianScaleSpace(200,350,4,3,-1,0.5,1.6,4.)
    op4 = bob.ip.GaussianScaleSpace(200,250,3,3,-1,0.5,1.6,4.)
    op5 = bob.ip.GaussianScaleSpace(200,250,4,4,-1,0.5,1.6,4.)
    op6 = bob.ip.GaussianScaleSpace(200,250,4,3,0,0.5,1.6,4.)
    op7 = bob.ip.GaussianScaleSpace(200,250,4,3,-1,0.75,1.6,4.)
    op8 = bob.ip.GaussianScaleSpace(200,250,4,3,-1,0.5,1.8,4.)
    op9 = bob.ip.GaussianScaleSpace(200,250,4,3,-1,0.5,1.6,3.)
    self.assertEqual(op1 == op1, True)
    self.assertEqual(op1 == op1b, True)
    self.assertEqual(op1 == op2, False)
    self.assertEqual(op1 == op3, False)
    self.assertEqual(op1 == op4, False)
    self.assertEqual(op1 == op5, False)
    self.assertEqual(op1 == op6, False)
    self.assertEqual(op1 == op7, False)
    self.assertEqual(op1 == op8, False)
    self.assertEqual(op1 == op9, False)
    self.assertEqual(op1 != op1, False)
    self.assertEqual(op1 != op1b, False)
    self.assertEqual(op1 != op2, True)
    self.assertEqual(op1 != op3, True)
    self.assertEqual(op1 != op4, True)
    self.assertEqual(op1 != op5, True)
    self.assertEqual(op1 != op6, True)
    self.assertEqual(op1 != op7, True)
    self.assertEqual(op1 != op8, True)
    self.assertEqual(op1 != op9, True)
