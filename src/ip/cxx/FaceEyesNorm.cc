/**
 * @file ip/cxx/FaceEyesNorm.cc
 * @date Thu Apr 14 21:03:45 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to perform geometric normalization.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/ip/FaceEyesNorm.h"

bob::ip::FaceEyesNorm::FaceEyesNorm( const double eyes_distance,
    const size_t crop_height, const size_t crop_width, const double crop_offset_h,
    const double crop_offset_w):
  m_eyes_distance(eyes_distance), m_eyes_angle(0.),
  m_crop_height(crop_height), m_crop_width(crop_width),
  m_crop_offset_h(crop_offset_h), m_crop_offset_w(crop_offset_w),
  m_out_shape(crop_height, crop_width),
  m_geom_norm(new GeomNorm(0., 0., crop_height, crop_width, crop_offset_h, crop_offset_w) ),
  m_cache_angle(0.), m_cache_scale(0.)
{
}

bob::ip::FaceEyesNorm::FaceEyesNorm(
    const unsigned crop_height, const unsigned crop_width,
    const unsigned re_y, const unsigned re_x,
    const unsigned le_y, const unsigned le_x)
:
  m_crop_height(crop_height),
  m_crop_width(crop_width),
  m_out_shape(crop_height, crop_width),
  m_cache_angle(0.), m_cache_scale(0.)
{
  double dy = (double)re_y - (double)le_y, dx = (double)re_x - (double)le_x;
  m_eyes_distance = std::sqrt(dx * dx + dy * dy);
  m_eyes_angle = getAngleToHorizontal(re_y, re_x, le_y, le_x);
  m_crop_offset_h = (re_y + le_y) / 2.;
  m_crop_offset_w = (re_x + le_x) / 2.;

  m_geom_norm = boost::shared_ptr<GeomNorm>(new GeomNorm(0., 0., crop_height, crop_width, m_crop_offset_h, m_crop_offset_w));
}


bob::ip::FaceEyesNorm::FaceEyesNorm( const FaceEyesNorm& other):
  m_eyes_distance(other.m_eyes_distance), m_eyes_angle(other.m_eyes_angle),
  m_crop_height(other.m_crop_height), m_crop_width(other.m_crop_width),
  m_crop_offset_h(other.m_crop_offset_h), m_crop_offset_w(other.m_crop_offset_w),
  m_out_shape(other.m_crop_height, other.m_crop_width),
  m_geom_norm(new GeomNorm(0., 0., m_crop_height, m_crop_width, m_crop_offset_h, m_crop_offset_w) )
{
}

bob::ip::FaceEyesNorm& 
bob::ip::FaceEyesNorm::operator=(const bob::ip::FaceEyesNorm& other)
{
  if (this != &other)
  {
    m_eyes_distance = other.m_eyes_distance;
    m_eyes_angle = other.m_eyes_angle;
    m_crop_height = other.m_crop_height;
    m_crop_width = other.m_crop_width;
    m_crop_offset_h = other.m_crop_offset_h;
    m_crop_offset_w = other.m_crop_offset_w;
    m_out_shape(0) = m_crop_height;
    m_out_shape(1) = m_crop_width;
    m_geom_norm.reset(new GeomNorm(0., 0, m_crop_height, m_crop_width, 
      m_crop_offset_h, m_crop_offset_w) );
    m_cache_angle = other.m_cache_angle;
    m_cache_scale = other.m_cache_scale;
  }
  return *this;
}

bool 
bob::ip::FaceEyesNorm::operator==(const bob::ip::FaceEyesNorm& b) const
{
  return (this->m_eyes_distance == b.m_eyes_distance && this->m_crop_height == b.m_crop_height && 
          this->m_crop_width == b.m_crop_width && this->m_crop_offset_h == b.m_crop_offset_h && 
          this->m_crop_offset_w == b.m_crop_offset_w);
}

bool 
bob::ip::FaceEyesNorm::operator!=(const bob::ip::FaceEyesNorm& b) const
{
  return !(this->operator==(b));
}


