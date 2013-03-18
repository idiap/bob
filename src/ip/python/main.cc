/**
 * @file ip/python/main.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "bob/config.h"
#include "bob/core/python/ndarray.h"

void bind_ip_version();
void bind_ip_color();
void bind_ip_block();
void bind_ip_crop_shift();
void bind_ip_extrapolate_mask();
void bind_ip_flipflop();
void bind_ip_gamma_correction();
void bind_ip_integral();
void bind_ip_scale();
void bind_ip_shear();
void bind_ip_zigzag();
void bind_ip_rotate();
void bind_ip_flow();
void bind_ip_dctfeatures();
void bind_ip_gabor_wavelet_transform();
void bind_ip_geomnorm();
void bind_ip_faceeyesnorm();
void bind_ip_tantriggs();
void bind_ip_histogram();
void bind_ip_lbp_new();
void bind_ip_gaussian();
void bind_ip_gaussian_scale_space();
void bind_ip_wgaussian();
void bind_ip_msr();
void bind_ip_sqi();
void bind_ip_median();
void bind_ip_sobel();
void bind_ip_drawing();
void bind_ip_spatiotempgrad();
void bind_ip_hog();
void bind_ip_glcm_uint8();
void bind_ip_glcm_uint16();
void bind_ip_glcmprop();

#if WITH_VLFEAT
void bind_ip_vlsift();
void bind_ip_vldsift();
#endif

BOOST_PYTHON_MODULE(_ip) {

  bob::python::setup_python("bob image processing classes and sub-classes");

  bind_ip_version();
  bind_ip_color();
  bind_ip_block();
  bind_ip_crop_shift();
  bind_ip_extrapolate_mask();
  bind_ip_flipflop();
  bind_ip_gamma_correction();
  bind_ip_integral();
  bind_ip_scale();
  bind_ip_shear();
  bind_ip_zigzag();
  bind_ip_rotate();
  bind_ip_flow();
  bind_ip_dctfeatures();
  bind_ip_gabor_wavelet_transform();
  bind_ip_geomnorm();
  bind_ip_faceeyesnorm();
  bind_ip_tantriggs();
  bind_ip_histogram();
  bind_ip_lbp_new();
  bind_ip_gaussian();
  bind_ip_gaussian_scale_space();
  bind_ip_wgaussian();
  bind_ip_msr();
  bind_ip_sqi();
  bind_ip_median();
  bind_ip_sobel();
  bind_ip_drawing();
  bind_ip_spatiotempgrad();
  bind_ip_hog();
  bind_ip_glcm_uint8();
  bind_ip_glcm_uint16();
  bind_ip_glcmprop();

#if WITH_VLFEAT
  bind_ip_vlsift();
  bind_ip_vldsift();
#endif
}
