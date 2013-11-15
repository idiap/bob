/**
 * @file sp/cxx/DCT1D.cc
 * @date Thu Nov 14 18:15:49 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief 1D Discrete Cosine Transform using a 1D FFT
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/sp/DCT1D.h>
#include <cmath>
#include <bob/core/assert.h>
#include <bob/core/cast.h>
#include <bob/core/array_copy.h>
#include <boost/math/constants/constants.hpp>

bob::sp::DCT1DAbstract::DCT1DAbstract():
  bob::sp::DCT1DAbstract::DCT1DAbstract(1)
{
}

bob::sp::DCT1DAbstract::DCT1DAbstract(const size_t length):
  m_length(length),
  m_working_array(length)
{
  initNormFactors();
}

bob::sp::DCT1DAbstract::DCT1DAbstract(
    const bob::sp::DCT1DAbstract& other):
  m_length(other.m_length),
  m_working_array(other.m_length)
{
  initNormFactors();
}

bob::sp::DCT1DAbstract::~DCT1DAbstract()
{
}

bob::sp::DCT1DAbstract&
bob::sp::DCT1DAbstract::operator=(const DCT1DAbstract& other)
{
  if (this != &other) {
    m_length = other.m_length;
    m_working_array.resize(m_length);
    initWorkingArray();
    initNormFactors();
  }
  return *this;
}

bool bob::sp::DCT1DAbstract::operator==(const bob::sp::DCT1DAbstract& b) const
{
  return (this->m_length == b.m_length);
}

bool bob::sp::DCT1DAbstract::operator!=(const bob::sp::DCT1DAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::DCT1DAbstract::operator()(const blitz::Array<double,1>& src,
  blitz::Array<double,1>& dst) const
{
  // Check input, inclusive dimension
  bob::core::array::assertCZeroBaseContiguous(src);
  const blitz::TinyVector<int,1> shape(m_length);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::DCT1DAbstract::setLength(const size_t length)
{
  m_length = length;
  m_working_array.resize(length);
  initWorkingArray();
  initNormFactors();
}

void bob::sp::DCT1DAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1byl = sqrt(1./(double)m_length);
  m_sqrt_2byl = sqrt(2./(double)m_length);
}


bob::sp::DCT1D::DCT1D():
  bob::sp::DCT1D(1)
{
}

bob::sp::DCT1D::DCT1D(const size_t length):
  bob::sp::DCT1DAbstract(length),
  m_fft(2*length),
  m_buffer_1(2*length),
  m_buffer_2(2*length)
{
  initWorkingArray();
}

bob::sp::DCT1D::DCT1D(const bob::sp::DCT1D& other):
  bob::sp::DCT1DAbstract(other),
  m_fft(other.m_length),
  m_buffer_1(2*other.m_length),
  m_buffer_2(2*other.m_length)
{
  initWorkingArray();
}

bob::sp::DCT1D::~DCT1D()
{
}

bob::sp::DCT1D&
bob::sp::DCT1D::operator=(const DCT1D& other)
{
  if (this != &other) {
    bob::sp::DCT1DAbstract::operator=(other);
    m_fft.setLength(other.m_length);
    m_buffer_1.resize(2*other.m_length);
    m_buffer_2.resize(2*other.m_length);
  }
  return *this;
}

void bob::sp::DCT1D::setLength(const size_t length)
{
  bob::sp::DCT1DAbstract::setLength(length);
  m_fft.setLength(2*m_length);
  m_buffer_1.resize(2*length);
  m_buffer_2.resize(2*length);
}

void bob::sp::DCT1D::processNoCheck(const blitz::Array<double,1>& src,
  blitz::Array<double,1>& dst) const
{
  blitz::Range r1 = blitz::Range(0,m_length-1);
  blitz::Array<std::complex<double>,1> m_b1 = m_buffer_1(r1);
  blitz::Array<std::complex<double>,1> m_b2 = m_buffer_2(r1);
  // Compute the DCT
  // 1. Make m_buffer_1 = [src 0]
  m_buffer_1 = 0.;
  m_b1 = src;
  // 2. Compute m_buffer_2 = fft(m_buffer_1)
  m_fft(m_buffer_1, m_buffer_2);
  // 3. Multiply: m_buffer_2(0:L-1) * exp(-J*PI*k/(2*L))
  m_b2 *= m_working_array;
  // 4. Take Real part of m_buffer_2(0:L-1)
  dst = blitz::real(m_b2(r1));
  // 5. Customized normalization factors:
  //      sqrt(1/L) for index 0
  dst(0) *= m_sqrt_1byl;
  //      sqrt(2/L) for index >0
  if (dst.extent(0) > 1) {
    blitz::Range r_dst(1,m_length-1);
    dst(r_dst) *= m_sqrt_2byl;
  }
}

void bob::sp::DCT1D::initWorkingArray()
{
  std::complex<double> J(0., 1.);
  const double PI = boost::math::constants::pi<double>();
  std::complex<double> factor = -J*PI / (double)(2*m_length);
  for (int i=0; i<(int)m_length; ++i)
    m_working_array(i) = exp(factor*(std::complex<double>)i);
}


bob::sp::IDCT1D::IDCT1D():
  bob::sp::IDCT1D(1)
{
}

bob::sp::IDCT1D::IDCT1D(const size_t length):
  bob::sp::DCT1DAbstract(length),
  m_ifft(length),
  m_buffer_1(length),
  m_buffer_2(length)
{
  initWorkingArray();
}

bob::sp::IDCT1D::IDCT1D(const bob::sp::IDCT1D& other):
  bob::sp::DCT1DAbstract(other),
  m_ifft(other.m_length),
  m_buffer_1(other.m_length),
  m_buffer_2(other.m_length)
{
  initWorkingArray();
}

bob::sp::IDCT1D::~IDCT1D()
{
}

bob::sp::IDCT1D&
bob::sp::IDCT1D::operator=(const IDCT1D& other)
{
  if (this != &other) {
    bob::sp::DCT1DAbstract::operator=(other);
    m_ifft.setLength(other.m_length);
    m_buffer_1.resize(other.m_length);
    m_buffer_2.resize(other.m_length);
  }
  return *this;
}


void bob::sp::IDCT1D::setLength(const size_t length)
{
  bob::sp::DCT1DAbstract::setLength(length);
  m_ifft.setLength(length);
  m_buffer_1.resize(length);
  m_buffer_2.resize(length);
}

void bob::sp::IDCT1D::processNoCheck(const blitz::Array<double,1>& src,
  blitz::Array<double,1>& dst) const
{
  // Compute the DCT
  // 1. Make m_buffer_1 = src*m_working_array
  m_buffer_1 = src * m_working_array;
  // 2. Compute m_buffer_2 = ifft(m_buffer_1)
  m_ifft(m_buffer_1, m_buffer_2);
  // 3. Take Real part of m_buffer_2
  m_buffer_2 = 2*blitz::real(m_buffer_2);
  // 4. Take the output:
  for(int i=0; i<(int)(m_length/2); ++i) {
    dst(2*i) = bob::core::cast<double>(m_buffer_2(i));
    dst(2*i+1) = bob::core::cast<double>(m_buffer_2(m_length-1-i));
  }
  if ((m_length % 2) == 1)
    dst(m_length-1) = bob::core::cast<double>(m_buffer_2(m_length/2));
}

void bob::sp::IDCT1D::initWorkingArray()
{
  std::complex<double> J(0., 1.);
  const double PI = boost::math::constants::pi<double>();
  std::complex<double> factor = J*PI / (double)(2*m_length);
  for (int i=0; i<(int)m_length; ++i)
    m_working_array(i) = exp(factor*(std::complex<double>)i) * sqrt((double)(m_length) / 2.);
  m_working_array(0) /= sqrt(2);
}
