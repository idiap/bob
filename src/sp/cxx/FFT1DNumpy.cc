/**
 * @file sp/cxx/FFT1DNumpy.cc
 * @date Fri Nov 15 09:32:32 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Fast Fourier Transform
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/sp/FFT1DNumpy.h>
#include <bob/core/assert.h>
#include <bob/core/array_copy.h>

bob::sp::FFT1DNumpyAbstract::FFT1DNumpyAbstract(const size_t length):
  m_length(length), m_wsave(4*length+15), m_buffer(2*length)
{
  initWorkingArray();
}

bob::sp::FFT1DNumpyAbstract::FFT1DNumpyAbstract(
    const bob::sp::FFT1DNumpyAbstract& other):
  m_length(other.m_length), m_buffer(2*other.m_length)
{
  m_wsave = bob::core::array::ccopy(other.m_wsave);
}

bob::sp::FFT1DNumpyAbstract::~FFT1DNumpyAbstract()
{
}

bob::sp::FFT1DNumpyAbstract&
bob::sp::FFT1DNumpyAbstract::operator=(const FFT1DNumpyAbstract& other)
{
  if (this != &other) {
    m_length = other.m_length;
    m_wsave = bob::core::array::ccopy(other.m_wsave);
    m_buffer.resize(2*other.m_length);
  }
  return *this;
}

bool bob::sp::FFT1DNumpyAbstract::operator==(const bob::sp::FFT1DNumpyAbstract& b) const
{
  return (this->m_length == b.m_length);
}

bool bob::sp::FFT1DNumpyAbstract::operator!=(const bob::sp::FFT1DNumpyAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::FFT1DNumpyAbstract::operator()(const blitz::Array<std::complex<double>,1>& src,
  blitz::Array<std::complex<double>,1>& dst) const
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

void bob::sp::FFT1DNumpyAbstract::setLength(const size_t length)
{
  m_length = length;
  m_wsave.resize(4*length+15);
  initWorkingArray();
  m_buffer.resize(2*length);
}

void bob::sp::FFT1DNumpyAbstract::initWorkingArray()
{
  double *wsave_ptr = m_wsave.data();
  cffti((int)m_length, wsave_ptr);
}


bob::sp::FFT1DNumpy::FFT1DNumpy(const size_t length):
  bob::sp::FFT1DNumpyAbstract(length)
{
}

bob::sp::FFT1DNumpy::FFT1DNumpy(const bob::sp::FFT1DNumpy& other):
  bob::sp::FFT1DNumpyAbstract(other)
{
}

bob::sp::FFT1DNumpy::~FFT1DNumpy()
{
}

bob::sp::FFT1DNumpy&
bob::sp::FFT1DNumpy::operator=(const FFT1DNumpy& other)
{
  if (this != &other) {
    bob::sp::FFT1DNumpyAbstract::operator=(other);
  }
  return *this;
}

void bob::sp::FFT1DNumpy::setLength(const size_t length)
{
  bob::sp::FFT1DNumpyAbstract::setLength(length);
}

void bob::sp::FFT1DNumpy::processNoCheck(const blitz::Array<std::complex<double>,1>& src,
  blitz::Array<std::complex<double>,1>& dst) const
{
  // Compute the FFT
  blitz::Range r1(0, 2*m_length-2, 2);
  blitz::Range r2(1, 2*m_length-1, 2);
  m_buffer(r1) = blitz::real(src);
  m_buffer(r2) = blitz::imag(src);
  double *buf_ptr = m_buffer.data();
  double *wsave_ptr = const_cast<double*>(m_wsave.data());
  cfftf(m_length, buf_ptr, wsave_ptr);
  dst = m_buffer(r1) + std::complex<double>(0.,1.) * m_buffer(r2);
}


bob::sp::IFFT1DNumpy::IFFT1DNumpy(const size_t length):
  bob::sp::FFT1DNumpyAbstract(length)
{
}

bob::sp::IFFT1DNumpy::IFFT1DNumpy(const bob::sp::IFFT1DNumpy& other):
  bob::sp::FFT1DNumpyAbstract(other)
{
}

bob::sp::IFFT1DNumpy::~IFFT1DNumpy()
{
}

bob::sp::IFFT1DNumpy&
bob::sp::IFFT1DNumpy::operator=(const IFFT1DNumpy& other)
{
  if (this != &other) {
    bob::sp::FFT1DNumpyAbstract::operator=(other);
  }
  return *this;
}


void bob::sp::IFFT1DNumpy::setLength(const size_t length)
{
  bob::sp::FFT1DNumpyAbstract::setLength(length);
}

void bob::sp::IFFT1DNumpy::processNoCheck(const blitz::Array<std::complex<double>,1>& src,
  blitz::Array<std::complex<double>,1>& dst) const
{
  // Compute the FFT
  blitz::Range r1(0, 2*m_length-2, 2);
  blitz::Range r2(1, 2*m_length-1, 2);
  m_buffer(r1) = blitz::real(src);
  m_buffer(r2) = blitz::imag(src);
  double *buf_ptr = m_buffer.data();
  double *wsave_ptr = const_cast<double*>(m_wsave.data());
  cfftb(m_length, buf_ptr, wsave_ptr);
  dst = m_buffer(r1) + std::complex<double>(0.,1.) * m_buffer(r2);
  dst /= (double)m_length;
}
