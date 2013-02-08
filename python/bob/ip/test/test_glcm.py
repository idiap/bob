#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Ivana Chingovska <ivana.chingovska@idiap.ch>
#Sun Jan 27 18:59:39 CET 2013
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

"""Tests the GLCM framework.
"""

import os, sys
import unittest
import bob
import math
import numpy
import pkg_resources

IMG_3x3_A = numpy.array([ [0, 1, 0],
                          [0, 1, 1],
                          [0, 1, 0]], dtype='uint8')
                          
IMG_3x3_B = numpy.array([ [0, 32, 64],
                          [64, 96, 128],
                          [128, 160, 192]], dtype='uint8')  
                          
IMG_3x3_C = numpy.array([ [0, 32, 64],
                          [64, 96, 128],
                          [128, 164, 201]], dtype='uint16')  
                                                  
                 
expected = numpy.array([[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]], dtype='double')                 
                 
                     
IMG_3x6_A = numpy.array([ [1, 1, 4, 2, 1, 0],
                          [2, 3, 3, 4, 0, 2],
                          [0, 2, 3, 1, 4, 4]], dtype='uint8')
    
                          
class GLCMTest(unittest.TestCase):
  """Performs various tests on the GLCM functionalities"""
  
  def test01_GLCM(self):
    # Test the computation of GLCM matrix
    glcm = bob.ip.GLCM('uint8')
    self.assertTrue(glcm.get_glcm_shape() == (256, 256, 1))
    res = glcm(IMG_3x3_A)    
    self.assertTrue( (res[0:2, 0:2, 0] == numpy.array([[0,3],[2,1]], dtype='float64')).all())
    
    glcm.offset = numpy.array([[1,0],[0,-1]], dtype='int32')    
    self.assertTrue(glcm.get_glcm_shape() == (256, 256, 2))
    res = glcm(IMG_3x3_A)
    self.assertTrue( (res[0:2, 0:2, 0] == numpy.array([[0,3],[2,1]], dtype='float64')).all())
    self.assertTrue( (res[0:2, 0:2, 1] == numpy.array([[2,1],[1,2]], dtype='float64')).all())
    
    glcm.symmetric = True
    res = glcm(IMG_3x3_A)
    self.assertTrue( (res[0:2, 0:2, 0] == numpy.array([[0,5],[5,2]], dtype='float64')).all())
    self.assertTrue( (res[0:2, 0:2, 1] == numpy.array([[4,2],[2,4]], dtype='float64')).all())
      
    glcm.symmetric = True
    glcm.normalized = True
    res = glcm(IMG_3x3_A)
    self.assertTrue( (res[0:2, 0:2, 0] == numpy.array([[0,5./12],[5./12,2./12]], dtype='float64')).all())
    self.assertTrue( (res[0:2, 0:2, 1] == numpy.array([[4./12,2./12],[2./12,4./12]], dtype='float64')).all())
    
    # now we check the results for the most common offsets (angles of 0, 45, 90 and 135 degrees)
    glcm.normalized = False
    glcm.symmetric = False
    glcm.offset = numpy.array([[1,0],[1, -1],[0,-1],[-1,-1]], dtype='int32')
    res = glcm(IMG_3x3_A)
    self.assertTrue( (res[0:2, 0:2, 0] == numpy.array([[0,3],[2,1]], dtype='float64')).all())
    self.assertTrue( (res[0:2, 0:2, 1] == numpy.array([[0,2],[1,1]], dtype='float64')).all())
    self.assertTrue( (res[0:2, 0:2, 2] == numpy.array([[2,1],[1,2]], dtype='float64')).all())
    self.assertTrue( (res[0:2, 0:2, 3] == numpy.array([[0,1],[2,1]], dtype='float64')).all())
  
  
  def test02_GLCM(self):
    # Additional tests on the computation of GLCM matrix    
    glcm = bob.ip.GLCM('uint8', num_levels=8);
    self.assertTrue(glcm.get_glcm_shape() == (8, 8, 1))
    self.assertTrue( (glcm.quantization_table == numpy.array([0,32,64,96,128,160,192,224])).all() )
    res = glcm(IMG_3x3_B)  
    self.assertTrue( (res[:,:,0] == expected).all())
    
  
  def test03_GLCM(self):
    # Additional tests on the computation of GLCM matrix  
    glcm = bob.ip.GLCM('uint16', numpy.array([0,19,55,92,128,164,201,237], dtype='uint16')) # thresholds selected according to Matlab quantization
    self.assertTrue(glcm.get_glcm_shape() == (8, 8, 1))
    self.assertEqual(glcm.num_levels,8)
    self.assertEqual(glcm.max_level,65535)
    self.assertEqual(glcm.min_level,0)
    res = glcm(IMG_3x3_C)
    self.assertTrue( (res[:,:,0] == expected).all())
    
 
    
  def test04_GLCM(self):
    # Test GLCM properties
    # The testing of the properties tests whether the results are compatible with the code given in http://www.mathworks.com/matlabcentral/fileexchange/22354-glcmfeatures4-m-vectorized-version-of-glcmfeatures1-m-with-code-changes. However, the indexing of the arrays there starts from 1 and in Bob it starts from 0. To avoid the descrepencies, some changes in that code is needed, in particluar in the i_matrix and j_matrix variables, as well as xplusy_index 
    glcm_prop = bob.ip.GLCMProp()
    glcm_matrix = numpy.array([[[0,0,2,0,0],[1,1,0,0,2],[0,1,0,2,0],[0,1,0,1,1],[1,0,1,0,1]]], dtype='double')
    res_matrix = numpy.ndarray((5,5,1), 'double')
    res_matrix[:,:,0] = glcm_matrix
    
    self.assertTrue(numpy.allclose(glcm_prop.angular_second_moment(res_matrix), [0.09333333])) # energy in [5],[6]
    self.assertTrue(numpy.allclose(glcm_prop.energy(res_matrix), [0.305505])) # doesn't exist in [6]
    self.assertTrue(numpy.allclose(glcm_prop.contrast(res_matrix), [3.66666666666667])) # contrast in [5],[6] 
    self.assertTrue(numpy.allclose(glcm_prop.variance(res_matrix), [5.90293333333333])) # sum of squares (variance) in [5]
    self.assertTrue(numpy.allclose(glcm_prop.auto_correlation(res_matrix), [4.73333333])) # auto-correlation in [5]
    self.assertTrue(numpy.allclose(glcm_prop.correlation(res_matrix), [0.0262698])) # correlation-p in [5]
    self.assertTrue(numpy.allclose(glcm_prop.correlation_m(res_matrix), [0.0262698])) # correlation-m in [5]
    self.assertTrue(numpy.allclose(glcm_prop.inv_diff_mom(res_matrix), [0.4372549])) # homogeneity-p in [5]
    self.assertTrue(numpy.allclose(glcm_prop.sum_avg(res_matrix), [4.33333333333333])) # sum average in [5] 
    self.assertTrue(numpy.allclose(glcm_prop.sum_var(res_matrix), [9.57993445])) # sum variance in [5]
    self.assertTrue(numpy.allclose(glcm_prop.sum_entropy(res_matrix), [1.93381])) # sum entropy in [5]
    self.assertTrue(numpy.allclose(glcm_prop.entropy(res_matrix), [2.43079132887823])) # entropy in [5]
    self.assertTrue(numpy.allclose(glcm_prop.dissimilarity(res_matrix), [1.53333333])) # dissimilarity in [5]
    self.assertTrue(numpy.allclose(glcm_prop.homogeneity(res_matrix), [0.50222222])) # homogeneity-m in [5]
    self.assertTrue(numpy.allclose(glcm_prop.diff_entropy(res_matrix), [1.48975032])) # difference entropy in [5]
    self.assertTrue(numpy.allclose(glcm_prop.diff_var(res_matrix), [3.66666667])) # difference variance in [5]
    self.assertTrue(numpy.allclose(glcm_prop.cluster_prom(res_matrix), [30.87407407])) # cluster prominance in [5]
    self.assertTrue(numpy.allclose(glcm_prop.cluster_shade(res_matrix), [0.07407407])) # cluster shade in [5]
    self.assertTrue(numpy.allclose(glcm_prop.max_prob(res_matrix), [0.13333333])) # maximum probability in [5]
    self.assertTrue(numpy.allclose(glcm_prop.inf_meas_corr1(res_matrix), [-0.46810262])) # information measure correlation 1 in [5]
    self.assertTrue(numpy.allclose(glcm_prop.inf_meas_corr2(res_matrix), [0.87955875])) # information measure correlation 2 in [5]
    self.assertTrue(numpy.allclose(glcm_prop.inv_diff(res_matrix), [0.50222222])) # inverse difference in [5]
    self.assertTrue(numpy.allclose(glcm_prop.inv_diff_norm(res_matrix), [ 0.788624338624339])) # inverse difference normalized in [5]
    self.assertTrue(numpy.allclose(glcm_prop.inv_diff_mom_norm(res_matrix), [0.8890875])) # inverse difference moment normalized in [5]
  
    self.assertTrue(numpy.allclose(glcm_prop.properties_by_name(res_matrix, ["angular second moment"]), numpy.array([0.09333333]))) # energy in [5],[6]
    
    
