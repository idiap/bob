#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sun Sep 18 18:16:50 2012 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

"""Tests the SIFT features extractor
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

class SIFT(unittest.TestCase):
  """Performs various tests for the bob.ip.SIFT class"""

  def test01_parametrization(self):
    # Parametrization tests
    op = bob.ip.SIFT(200,250,4,3,-1,0.5,1.6,4.)
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
    cont_t = 0.03
    edge_t = 10.
    norm_t = 0.2
    f=4.
    op = bob.ip.SIFT(A.shape[0],A.shape[1],No,Ns,0,sigma_n,sigma0,cont_t,edge_t,norm_t,f,bob.sp.BorderType.NearestNeighbour)
    kp=[bob.ip.GSSKeypoint(1.6,326,270)]
    B=op.compute_descriptor(A,kp)
    C=B[0]
    #bob.io.save(C, F(os.path.join("sift","vlimg_ref_cmp.hdf5"))) # Generated using initial bob version
    C_ref = bob.io.load(F(os.path.join("sift", "vlimg_ref_cmp.hdf5")))
    self.assertTrue( numpy.allclose(C, C_ref, 1e-5, 1e-5) )
    """
    Descriptor returned by vlfeat 0.9.14. 
      Differences with our implementation are (but not limited to):
      - Single vs. double precision (with error propagation in the Gaussian pyramid)
      
      0          0         0          0.290434    65.2558   62.7004   62.6646    0.557657 
      0.592095   0.145797  0          0.00843264 127.977    56.457     7.54261   0.352965
     97.3214     9.24475   0          0.0204793   50.0755   12.69      1.2646   20.525
     91.3951     8.68794   0.232415   0.688901     7.03954   6.8892    8.55246  41.1051
      0.0116815  0.342656  2.76365    0.350923     3.48516  29.5739  127.977    28.5115
      5.92045    4.61406   1.16143    0.00232113  45.9274   90.237   127.977    21.7975
    116.967     68.2782    0.278292   0.000890405 20.5523   23.5499    5.12068  14.6013
     63.4585    69.2397   18.4443    18.6347       7.60615   4.41878   5.29352  19.1335 
      0.0283694 11.3307  127.977     16.1103       0.351831  0.762431 51.0464   13.5331 
     10.6187    71.1094  127.977      6.76088      0.157741  3.84676  40.6852   23.2877 
    127.977    115.818    43.3812     7.07351      0.242382  1.60356   2.59673   2.55512 
     96.3921    39.6973    8.31371   16.4943      17.4623    1.30552   0.224244  1.14927 
      7.40859   13.8157  127.977     25.6779       8.35931   9.28288   1.93504   1.90398 
      6.50493   26.9885  127.977     32.5336      16.6373    8.03625   0.242855  0.791766 
     44.7504    20.7554   35.8107    34.2561      26.2423   10.6024    2.14291  12.8046 
     54.9029     2.88965   0.0166734  0.227938    18.4405    6.35371   3.85071  28.1302
    """

  def test03_comparison(self):
    # Comparisons tests
    op1 = bob.ip.SIFT(200,250,4,3,-1,0.5,1.6,4.)
    op1b = bob.ip.SIFT(200,250,4,3,-1,0.5,1.6,4.)
    op2 = bob.ip.SIFT(300,250,4,3,-1,0.5,1.6,4.)
    op3 = bob.ip.SIFT(200,350,4,3,-1,0.5,1.6,4.)
    op4 = bob.ip.SIFT(200,250,3,3,-1,0.5,1.6,4.)
    op5 = bob.ip.SIFT(200,250,4,4,-1,0.5,1.6,4.)
    op6 = bob.ip.SIFT(200,250,4,3,0,0.5,1.6,4.)
    op7 = bob.ip.SIFT(200,250,4,3,-1,0.75,1.6,4.)
    op8 = bob.ip.SIFT(200,250,4,3,-1,0.5,1.8,4.)
    op9 = bob.ip.SIFT(200,250,4,3,-1,0.5,1.6,3.)
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
