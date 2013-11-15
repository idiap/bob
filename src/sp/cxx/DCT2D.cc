/**
 * @file sp/cxx/DCT2D.cc
 * @date Thu Nov 14 22:59:47 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a 2D Discrete Cosine Transform using
 * a 1D DCT implementation.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/sp/DCT2D.h>
#include <bob/core/assert.h>

bob::sp::DCT2DAbstract::DCT2DAbstract():
  bob::sp::DCT2DAbstract::DCT2DAbstract(1, 1)
{
}

bob::sp::DCT2DAbstract::DCT2DAbstract(
    const size_t height, const size_t width):
  m_height(height), m_width(width),
  m_buffer_hw(height, width), m_buffer_h(height),
  m_buffer_h2(height)
{
}

bob::sp::DCT2DAbstract::DCT2DAbstract(
    const bob::sp::DCT2DAbstract& other):
  m_height(other.m_height), m_width(other.m_width),
  m_buffer_hw(other.m_height, other.m_width), m_buffer_h(other.m_height),
  m_buffer_h2(other.m_height)
{
}

bob::sp::DCT2DAbstract::~DCT2DAbstract()
{
}

bob::sp::DCT2DAbstract&
bob::sp::DCT2DAbstract::operator=(const DCT2DAbstract& other)
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

bool bob::sp::DCT2DAbstract::operator==(const bob::sp::DCT2DAbstract& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width);
}

bool bob::sp::DCT2DAbstract::operator!=(const bob::sp::DCT2DAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::DCT2DAbstract::operator()(const blitz::Array<double,2>& src,
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

void bob::sp::DCT2DAbstract::setHeight(const size_t height)
{
  m_height = height;
  m_buffer_hw.resize(m_height, m_width);
  m_buffer_h.resize(m_height);
  m_buffer_h2.resize(m_height);
}

void bob::sp::DCT2DAbstract::setWidth(const size_t width)
{
  m_width = width;
  m_buffer_hw.resize(m_height, m_width);
}

void bob::sp::DCT2DAbstract::setShape(const size_t height, const size_t width)
{
  m_height = height;
  m_width = width;
  m_buffer_hw.resize(m_height, m_width);
  m_buffer_h.resize(m_height);
  m_buffer_h2.resize(m_height);
}


bob::sp::DCT2D::DCT2D():
  bob::sp::DCT2D::DCT2D(1, 1)
{
}

bob::sp::DCT2D::DCT2D(const size_t height, const size_t width):
  bob::sp::DCT2DAbstract::DCT2DAbstract(height, width),
  m_dct_h(height),
  m_dct_w(width)
{
}

bob::sp::DCT2D::DCT2D(const bob::sp::DCT2D& other):
  bob::sp::DCT2DAbstract(other),
  m_dct_h(other.m_height),
  m_dct_w(other.m_width)
{
}

bob::sp::DCT2D::~DCT2D()
{
}

bob::sp::DCT2D&
bob::sp::DCT2D::operator=(const DCT2D& other)
{
  if (this != &other) {
    bob::sp::DCT2DAbstract::operator=(other);
    m_dct_h.setLength(other.m_height);
    m_dct_w.setLength(other.m_width);
  }
  return *this;
}

void bob::sp::DCT2D::setHeight(const size_t height)
{
  bob::sp::DCT2DAbstract::setHeight(height);
  m_dct_h.setLength(height);
}

void bob::sp::DCT2D::setWidth(const size_t width)
{
  bob::sp::DCT2DAbstract::setWidth(width);
  m_dct_w.setLength(width);
}

void bob::sp::DCT2D::setShape(const size_t height, const size_t width)
{
  bob::sp::DCT2DAbstract::setShape(height, width);
  m_dct_h.setLength(height);
  m_dct_w.setLength(width);
}

void bob::sp::DCT2D::processNoCheck(const blitz::Array<double,2>& src,
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


bob::sp::IDCT2D::IDCT2D():
  bob::sp::IDCT2D::IDCT2D(1, 1)
{
}

bob::sp::IDCT2D::IDCT2D(const size_t height, const size_t width):
  bob::sp::DCT2DAbstract::DCT2DAbstract(height, width),
  m_idct_h(height),
  m_idct_w(width)
{
}

bob::sp::IDCT2D::IDCT2D(const bob::sp::IDCT2D& other):
  bob::sp::DCT2DAbstract(other),
  m_idct_h(other.m_height),
  m_idct_w(other.m_width)
{
}

bob::sp::IDCT2D::~IDCT2D()
{
}

bob::sp::IDCT2D&
bob::sp::IDCT2D::operator=(const IDCT2D& other)
{
  if (this != &other) {
    bob::sp::DCT2DAbstract::operator=(other);
    m_idct_h.setLength(other.m_height);
    m_idct_w.setLength(other.m_width);
  }
  return *this;
}

void bob::sp::IDCT2D::setHeight(const size_t height)
{
  bob::sp::DCT2DAbstract::setHeight(height);
  m_idct_h.setLength(height);
}

void bob::sp::IDCT2D::setWidth(const size_t width)
{
  bob::sp::DCT2DAbstract::setWidth(width);
  m_idct_w.setLength(width);
}

void bob::sp::IDCT2D::setShape(const size_t height, const size_t width)
{
  bob::sp::DCT2DAbstract::setShape(height, width);
  m_idct_h.setLength(height);
  m_idct_w.setLength(width);
}

void bob::sp::IDCT2D::processNoCheck(const blitz::Array<double,2>& src,
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
