/**
 * @file src/cxx/ip/src/GeomNorm.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to perform geometric normalization.
 */

#include "ip/GeomNorm.h"

namespace ip = Torch::ip;

ip::GeomNorm::GeomNorm( const int eyes_distance, const int center_eyes_h, 
  const int center_eyes_w, const int height, const int width, 
  const int border_h, const int border_w):
  m_eyes_distance(eyes_distance), m_center_eyes_h(center_eyes_h),
  m_center_eyes_w(center_eyes_w), m_height(height), m_width(width),
  m_border_h(border_h), m_border_w(border_w), 
  m_out_shape(height+border_h*2,width+border_w*2)
{
}

ip::GeomNorm::~GeomNorm() { }

