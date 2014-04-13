/**
 * @file ip/cxx/BlockCellGradientDescriptors.cc
 * @date Sun Apr 22 19:55:44 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/ip/BlockCellGradientDescriptors.h"
#include "bob/core/assert.h"

bob::ip::GradientMaps::GradientMaps(const size_t height,
    const size_t width, const GradientMagnitudeType mag_type):
  m_gy(height, width), m_gx(height, width), m_mag_type(mag_type)
{
}

bob::ip::GradientMaps::GradientMaps(const bob::ip::GradientMaps& other):
  m_gy(other.m_gy.extent(0), other.m_gy.extent(1)),
  m_gx(other.m_gx.extent(0), other.m_gx.extent(1)),
  m_mag_type(other.m_mag_type)
{
}

bob::ip::GradientMaps&
bob::ip::GradientMaps::operator=(const bob::ip::GradientMaps& other)
{
  if (this != &other)
  {
    m_gy.resize(other.m_gy.extent(0), other.m_gy.extent(1));
    m_gx.resize(other.m_gx.extent(0), other.m_gx.extent(1));
    m_mag_type = other.m_mag_type;
  }
  return *this;
}

bool
bob::ip::GradientMaps::operator==(const bob::ip::GradientMaps& b) const
{
  return (this->m_gy.extent(0) == b.m_gy.extent(0) &&
          this->m_gy.extent(1) == b.m_gy.extent(1) &&
          this->m_gx.extent(0) == b.m_gx.extent(0) &&
          this->m_gx.extent(1) == b.m_gx.extent(1) &&
          this->m_mag_type == b.m_mag_type);
}

bool
bob::ip::GradientMaps::operator!=(const bob::ip::GradientMaps& b) const
{
  return !(this->operator==(b));
}

void bob::ip::GradientMaps::resize(const size_t height, const size_t width)
{
  m_gy.resize(height,width);
  m_gx.resize(height,width);
}

void bob::ip::GradientMaps::setHeight(const size_t height)
{
  m_gy.resize((int)height,m_gy.extent(1));
  m_gx.resize((int)height,m_gx.extent(1));
}

void bob::ip::GradientMaps::setWidth(const size_t width)
{
  m_gy.resize(m_gy.extent(0),(int)width);
  m_gx.resize(m_gx.extent(0),(int)width);
}

