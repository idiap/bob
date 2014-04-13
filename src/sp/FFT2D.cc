/**
 * @file sp/cxx/FFT2D.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 2D Fast Fourier Transform
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/sp/FFT2D.h>
#include <bob/core/assert.h>

bob::sp::FFT2DAbstract::FFT2DAbstract():
  m_height(1), m_width(1),
  m_buffer_hw(1,1), m_buffer_h(1), m_buffer_h2(1)
{
}

bob::sp::FFT2DAbstract::FFT2DAbstract(
    const size_t height, const size_t width):
  m_height(height), m_width(width),
  m_buffer_hw(height, width), m_buffer_h(height),
  m_buffer_h2(height)
{
  if (m_height < 1) 
    throw std::runtime_error("DCT height should be at least 1.");
  if (m_width < 1) 
    throw std::runtime_error("DCT width should be at least 1.");
}

bob::sp::FFT2DAbstract::FFT2DAbstract(
    const bob::sp::FFT2DAbstract& other):
  m_height(other.m_height), m_width(other.m_width),
  m_buffer_hw(other.m_height, other.m_width), m_buffer_h(other.m_height),
  m_buffer_h2(other.m_height)
{
}

bob::sp::FFT2DAbstract::~FFT2DAbstract()
{
}

bob::sp::FFT2DAbstract&
bob::sp::FFT2DAbstract::operator=(const FFT2DAbstract& other)
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

bool bob::sp::FFT2DAbstract::operator==(const bob::sp::FFT2DAbstract& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width);
}

bool bob::sp::FFT2DAbstract::operator!=(const bob::sp::FFT2DAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::FFT2DAbstract::operator()(const blitz::Array<std::complex<double>,2>& src,
  blitz::Array<std::complex<double>,2>& dst) const
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

void bob::sp::FFT2DAbstract::setHeight(const size_t height)
{
  if (height < 1) 
    throw std::runtime_error("DCT height should be at least 1.");
  m_height = height;
  m_buffer_hw.resize(m_height, m_width);
  m_buffer_h.resize(m_height);
  m_buffer_h2.resize(m_height);
}

void bob::sp::FFT2DAbstract::setWidth(const size_t width)
{
  if (width < 1) 
    throw std::runtime_error("DCT width should be at least 1.");
  m_width = width;
  m_buffer_hw.resize(m_height, m_width);
}

void bob::sp::FFT2DAbstract::setShape(const size_t height, const size_t width)
{
  if (height < 1) 
    throw std::runtime_error("DCT height should be at least 1.");
  if (width < 1) 
    throw std::runtime_error("DCT width should be at least 1.");
  m_height = height;
  m_width = width;
  m_buffer_hw.resize(height, width);
  m_buffer_h.resize(height);
  m_buffer_h2.resize(height);
}


bob::sp::FFT2D::FFT2D():
  bob::sp::FFT2DAbstract(1,1),
  m_fft_h(1), m_fft_w(1)
{
}

bob::sp::FFT2D::FFT2D(const size_t height, const size_t width):
  bob::sp::FFT2DAbstract(height, width),
  m_fft_h(height),
  m_fft_w(width)
{
}

bob::sp::FFT2D::FFT2D(const bob::sp::FFT2D& other):
  bob::sp::FFT2DAbstract(other),
  m_fft_h(other.m_height),
  m_fft_w(other.m_width)
{
}

bob::sp::FFT2D::~FFT2D()
{
}

bob::sp::FFT2D&
bob::sp::FFT2D::operator=(const FFT2D& other)
{
  if (this != &other) {
    bob::sp::FFT2DAbstract::operator=(other);
    m_fft_h.setLength(other.m_height);
    m_fft_w.setLength(other.m_width);
  }
  return *this;
}

void bob::sp::FFT2D::setHeight(const size_t height)
{
  bob::sp::FFT2DAbstract::setHeight(height);
  m_fft_h.setLength(height);
}

void bob::sp::FFT2D::setWidth(const size_t width)
{
  bob::sp::FFT2DAbstract::setWidth(width);
  m_fft_w.setLength(width);
}

void bob::sp::FFT2D::setShape(const size_t height, const size_t width)
{
  bob::sp::FFT2DAbstract::setShape(height, width);
  m_fft_h.setLength(height);
  m_fft_w.setLength(width);
}

void bob::sp::FFT2D::processNoCheck(const blitz::Array<std::complex<double>,2>& src,
  blitz::Array<std::complex<double>,2>& dst) const
{
  blitz::Range rall = blitz::Range::all();
  // Compute the FFT
  for (int i=0; i<(int)m_height; ++i) {
    const blitz::Array<std::complex<double>,1> srci = src(i, rall);
    blitz::Array<std::complex<double>,1> bufi = m_buffer_hw(i, rall);
    m_fft_w(srci, bufi);
  }
  for (int j=0; j<(int)m_width; ++j) {
    m_buffer_h = m_buffer_hw(rall, j);
    m_fft_h(m_buffer_h, m_buffer_h2);
    blitz::Array<std::complex<double>,1> dstj = dst(rall, j);
    dstj = m_buffer_h2;
  }
}


bob::sp::IFFT2D::IFFT2D():
  bob::sp::FFT2DAbstract(1,1),
  m_ifft_h(1), m_ifft_w(1)
{
}

bob::sp::IFFT2D::IFFT2D(const size_t height, const size_t width):
  bob::sp::FFT2DAbstract(height, width),
  m_ifft_h(height),
  m_ifft_w(width)
{
}

bob::sp::IFFT2D::IFFT2D(const bob::sp::IFFT2D& other):
  bob::sp::FFT2DAbstract(other),
  m_ifft_h(other.m_height),
  m_ifft_w(other.m_width)
{
}

bob::sp::IFFT2D::~IFFT2D()
{
}

bob::sp::IFFT2D&
bob::sp::IFFT2D::operator=(const IFFT2D& other)
{
  if (this != &other) {
    bob::sp::FFT2DAbstract::operator=(other);
    m_ifft_h.setLength(other.m_height);
    m_ifft_w.setLength(other.m_width);
  }
  return *this;
}

void bob::sp::IFFT2D::setHeight(const size_t height)
{
  bob::sp::FFT2DAbstract::setHeight(height);
  m_ifft_h.setLength(height);
}

void bob::sp::IFFT2D::setWidth(const size_t width)
{
  bob::sp::FFT2DAbstract::setWidth(width);
  m_ifft_w.setLength(width);
}

void bob::sp::IFFT2D::setShape(const size_t height, const size_t width)
{
  bob::sp::FFT2DAbstract::setShape(height, width);
  m_ifft_h.setLength(height);
  m_ifft_w.setLength(width);
}

void bob::sp::IFFT2D::processNoCheck(const blitz::Array<std::complex<double>,2>& src,
  blitz::Array<std::complex<double>,2>& dst) const
{
  blitz::Range rall = blitz::Range::all();
  // Compute the FFT
  for (int i=0; i<(int)m_height; ++i) {
    const blitz::Array<std::complex<double>,1> srci = src(i, rall);
    blitz::Array<std::complex<double>,1> bufi = m_buffer_hw(i, rall);
    m_ifft_w(srci, bufi);
  }
  for (int j=0; j<(int)m_width; ++j) {
    m_buffer_h = m_buffer_hw(rall, j);
    m_ifft_h(m_buffer_h, m_buffer_h2);
    blitz::Array<std::complex<double>,1> dstj = dst(rall, j);
    dstj = m_buffer_h2;
  }
}
