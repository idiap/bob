/**
 * @file sp/cxx/DCT2DNumpy.cc
 * @date Thu Nov 14 22:59:47 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a 2D Discrete Cosine Transform using
 * a 1D DCT implementation.
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

#include <bob/sp/DCT2DNumpy.h>
#include <bob/core/assert.h>

bob::sp::DCT2DNumpyAbstract::DCT2DNumpyAbstract(
    const size_t height, const size_t width):
  m_height(height), m_width(width),
  m_buffer_hw(height, width), m_buffer_h(height),
  m_buffer_h2(height)
{
}

bob::sp::DCT2DNumpyAbstract::DCT2DNumpyAbstract(
    const bob::sp::DCT2DNumpyAbstract& other):
  m_height(other.m_height), m_width(other.m_width),
  m_buffer_hw(other.m_height, other.m_width), m_buffer_h(other.m_height),
  m_buffer_h2(other.m_height)
{
}

bob::sp::DCT2DNumpyAbstract::~DCT2DNumpyAbstract()
{
}

bob::sp::DCT2DNumpyAbstract& 
bob::sp::DCT2DNumpyAbstract::operator=(const DCT2DNumpyAbstract& other)
{
  if (this != &other) {
    setHeight(other.m_height);
    setWidth(other.m_width);
    m_buffer_hw.resize(other.m_height, other.m_width);
    m_buffer_h.resize(other.m_height);
    m_buffer_h2.resize(other.m_height);
  }
  return *this;
}

bool bob::sp::DCT2DNumpyAbstract::operator==(const bob::sp::DCT2DNumpyAbstract& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width);
}

bool bob::sp::DCT2DNumpyAbstract::operator!=(const bob::sp::DCT2DNumpyAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::DCT2DNumpyAbstract::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst) const
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height,m_width);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape(dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::DCT2DNumpyAbstract::setHeight(const size_t height)
{
  m_height = height;
  m_buffer_hw.resize(m_height, m_width);
  m_buffer_h.resize(m_height);
  m_buffer_h2.resize(m_height);
}

void bob::sp::DCT2DNumpyAbstract::setWidth(const size_t width)
{
  m_width = width;
  m_buffer_hw.resize(m_height, m_width);
  m_buffer_h.resize(m_height);
  m_buffer_h2.resize(m_height);
}
 

bob::sp::DCT2DNumpy::DCT2DNumpy(const size_t height, const size_t width):
  bob::sp::DCT2DNumpyAbstract::DCT2DNumpyAbstract(height, width),
  m_dct_h(height),
  m_dct_w(width)
{
}

bob::sp::DCT2DNumpy::DCT2DNumpy(const bob::sp::DCT2DNumpy& other):
  bob::sp::DCT2DNumpyAbstract(other),
  m_dct_h(other.m_height),
  m_dct_w(other.m_width)
{
}

bob::sp::DCT2DNumpy::~DCT2DNumpy()
{
}

bob::sp::DCT2DNumpy& 
bob::sp::DCT2DNumpy::operator=(const DCT2DNumpy& other)
{
  if (this != &other) {
    bob::sp::DCT2DNumpyAbstract::operator=(other);
    m_dct_h.setLength(other.m_height);
    m_dct_w.setLength(other.m_width);
  }
  return *this;
}

void bob::sp::DCT2DNumpy::setHeight(const size_t height)
{
  bob::sp::DCT2DNumpyAbstract::setHeight(height);
  m_dct_h.setLength(height);
}

void bob::sp::DCT2DNumpy::setWidth(const size_t width)
{
  bob::sp::DCT2DNumpyAbstract::setWidth(width);
  m_dct_w.setLength(width);
}
 
void bob::sp::DCT2DNumpy::processNoCheck(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst) const
{
  blitz::Range rall = blitz::Range::all();
  // Compute the DCT
  for (int i=0; i<(int)m_height; ++i) {
    const blitz::Array<double,1> srci = src(i, rall);
    blitz::Array<double,1> bufi = m_buffer_hw(i, rall);
    m_dct_w(srci, bufi);
  }
  for (int j=0; j<(int)m_width; ++j) {
    m_buffer_h = m_buffer_hw(rall, j);
    m_dct_h(m_buffer_h, m_buffer_h2);
    blitz::Array<double,1> dstj = dst(rall, j);
    dstj = m_buffer_h2;
  }
}


bob::sp::IDCT2DNumpy::IDCT2DNumpy(const size_t height, const size_t width):
  bob::sp::DCT2DNumpyAbstract::DCT2DNumpyAbstract(height, width),
  m_idct_h(height),
  m_idct_w(width)
{
}

bob::sp::IDCT2DNumpy::IDCT2DNumpy(const bob::sp::IDCT2DNumpy& other):
  bob::sp::DCT2DNumpyAbstract(other),
  m_idct_h(other.m_height),
  m_idct_w(other.m_width)
{
}

bob::sp::IDCT2DNumpy::~IDCT2DNumpy()
{
}

bob::sp::IDCT2DNumpy& 
bob::sp::IDCT2DNumpy::operator=(const IDCT2DNumpy& other)
{
  if (this != &other) {
    bob::sp::DCT2DNumpyAbstract::operator=(other);
    m_idct_h.setLength(other.m_height);
    m_idct_w.setLength(other.m_width);
  }
  return *this;
}

void bob::sp::IDCT2DNumpy::setHeight(const size_t height)
{
  bob::sp::DCT2DNumpyAbstract::setHeight(height);
  m_idct_h.setLength(height);
}

void bob::sp::IDCT2DNumpy::setWidth(const size_t width)
{
  bob::sp::DCT2DNumpyAbstract::setWidth(width);
  m_idct_w.setLength(width);
}

void bob::sp::IDCT2DNumpy::processNoCheck(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst) const
{
  blitz::Range rall = blitz::Range::all();
  // Compute the DCT
  for (int i=0; i<(int)m_height; ++i) {
    const blitz::Array<double,1> srci = src(i, rall);
    blitz::Array<double,1> bufi = m_buffer_hw(i, rall);
    m_idct_w(srci, bufi);
  }
  for (int j=0; j<(int)m_width; ++j) {
    m_buffer_h = m_buffer_hw(rall, j);
    m_idct_h(m_buffer_h, m_buffer_h2);
    blitz::Array<double,1> dstj = dst(rall, j);
    dstj = m_buffer_h2;
  }
}
