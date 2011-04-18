/**
 * @file src/cxx/ip/src/GeomNorm.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to perform geometric normalization.
 */

#include "ip/GeomNorm.h"

namespace ip = Torch::ip;

ip::GeomNorm::GeomNorm( const double rotation_angle, const double scaling_factor,
  const int crop_height, const int crop_width, const int crop_offset_h, 
  const int crop_offset_w): 
  m_rotate(new Rotate(rotation_angle)), m_scaling_factor(scaling_factor),
  m_crop_height(crop_height), m_crop_width(crop_width),
  m_crop_offset_h(crop_offset_h), m_crop_offset_w(crop_offset_w),
  m_out_shape(crop_height, crop_width)
{
}

ip::GeomNorm::~GeomNorm() { }

