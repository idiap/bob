/**
 * @file src/cxx/ip/src/GeomNorm2.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to perform geometric normalization.
 */

#include "ip/GeomNorm2.h"

namespace ip = Torch::ip;

ip::GeomNormNew::GeomNormNew( const double rotation_angle, const int scaling_factor,
  const int crop_height, const int crop_width, const int crop_offset_h, 
  const int crop_offset_w): 
  m_rotation_angle(rotation_angle), m_scaling_factor(scaling_factor),
  m_crop_height(crop_height), m_crop_width(crop_width),
  m_crop_offset_h(crop_offset_h), m_crop_offset_w(crop_offset_w),
  m_out_shape(crop_height, crop_width)
{
}

ip::GeomNormNew::~GeomNormNew() { }

