.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Sun Apr 3 19:18:37 2011 +0200
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
.. 
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, version 3 of the License.
.. 
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
.. 
.. You should have received a copy of the GNU General Public License
.. along with this program.  If not, see <http://www.gnu.org/licenses/>.

.. Index file for the Python bob::ip bindings

==================
 Image Processing
==================

This module contains utilities for image processing.

.. module:: bob.ip

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   block
   crop
   draw_box
   draw_cross
   draw_cross_plus
   draw_line
   draw_point
   draw_point_
   extrapolate_mask
   flip
   flop
   flow_error
   gamma_correction
   get_angle_to_horizontal
   get_block_3d_output_shape
   get_block_4d_output_shape
   get_rotated_output_shape
   get_shear_x_shape
   get_shear_y_shape
   gray_to_rgb
   gray_to_rgb_f
   gray_to_rgb_u16
   gray_to_rgb_u8
   histogram
   histogram_
   histogram_equalization
   hog_compute_histogram
   hog_compute_histogram_
   hsl_to_rgb
   hsl_to_rgb_f
   hsl_to_rgb_u16
   hsl_to_rgb_u8
   hsv_to_rgb
   hsv_to_rgb_f
   hsv_to_rgb_u16
   hsv_to_rgb_u8
   integral
   laplacian_avg_hs
   laplacian_avg_hs_opencv
   max_rect_in_mask
   normalize_block
   normalize_block_
   normalize_gabor_jet
   rgb_to_gray
   rgb_to_gray_f
   rgb_to_gray_u16
   rgb_to_gray_u8
   rgb_to_hsl
   rgb_to_hsl_f
   rgb_to_hsl_u16
   rgb_to_hsl_u8
   rgb_to_hsv
   rgb_to_hsv_f
   rgb_to_hsv_u16
   rgb_to_hsv_u8
   rgb_to_yuv
   rgb_to_yuv_f
   rgb_to_yuv_u16
   rgb_to_yuv_u8
   rotate
   scale
   scale_as
   shear_x
   shear_y
   shift
   try_draw_point
   yuv_to_rgb
   yuv_to_rgb_f
   yuv_to_rgb_u16
   yuv_to_rgb_u8
   zigzag

.. rubric:: Classes

.. autosummary::
   :toctree: generated/

   BlockNorm
   CentralGradient
   DCTFeatures
   ELBPType
   FaceEyesNorm
   ForwardGradient
   GLCM
   GLCMProp
   GSSKeypoint
   GSSKeypointInfo
   GaborKernel
   GaborWaveletTransform
   Gaussian
   GaussianScaleSpace
   GeomNorm
   GradientMagnitudeType
   GradientMaps
   HOG
   HornAndSchunckFlow
   HornAndSchunckGradient
   IsotropicGradient
   LBP
   LBPHSFeatures
   LBPTop
   Median_float64
   Median_uint16
   Median_uint8
   MultiscaleRetinex
   PrewittGradient
   RescaleAlgorithm
   RotateAlgorithm
   SIFT
   SelfQuotientImage
   Sobel
   SobelGradient
   TanTriggs
   VLDSIFT
   VLSIFT
   VanillaHornAndSchunckFlow
   WeightedGaussian
