/**
 * @file src/cxx/ip/src/FaceEyesNorm.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to perform geometric normalization.
 */

#include "ip/FaceEyesNorm.h"

namespace ip = Torch::ip;

ip::FaceEyesNorm::FaceEyesNorm( const int eyes_distance, 
  const int crop_height, const int crop_width, const int crop_offset_h,
  const int crop_offset_w):
  m_eyes_distance(eyes_distance), m_crop_height(crop_height),
  m_crop_width(crop_width), m_crop_offset_h(crop_offset_h),
  m_crop_offset_w(crop_offset_w), m_out_shape(crop_height, crop_width), 
  m_geom_norm(new GeomNorm(0., 0, crop_height, crop_width, crop_offset_h, 
    crop_offset_w) )
{
}

ip::FaceEyesNorm::~FaceEyesNorm() { }

