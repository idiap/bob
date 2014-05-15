/**
 * @date Fri Nov 15 09:32:32 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Fast Fourier Transform
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/sp/FFT1D.h>
#include <bob/core/assert.h>
#include <bob/core/array_copy.h>

bob::sp::FFT1DAbstract::FFT1DAbstract():
  m_length(1), m_wsave(4*1+15), m_buffer(2)
{
  initWorkingArray();
}

bob::sp::FFT1DAbstract::FFT1DAbstract(const size_t length):
  m_length(length), m_wsave(4*length+15), m_buffer(2*length)
{
  if (length < 1) 
    throw std::runtime_error("FFT length should be at least 1.");
  initWorkingArray();
}

bob::sp::FFT1DAbstract::FFT1DAbstract(
    const bob::sp::FFT1DAbstract& other):
  m_length(other.m_length), m_wsave(other.m_wsave.shape()),
  m_buffer(2*other.m_length)
{
  m_wsave = bob::core::array::ccopy(other.m_wsave);
}

bob::sp::FFT1DAbstract::~FFT1DAbstract()
{
}

bob::sp::FFT1DAbstract&
bob::sp::FFT1DAbstract::operator=(const FFT1DAbstract& other)
{
  if (this != &other) {
    m_length = other.m_length;
    m_wsave.resize(other.m_wsave.shape());
    m_wsave = bob::core::array::ccopy(other.m_wsave);
    m_buffer.resize(2*other.m_length);
  }
  return *this;
}

bool bob::sp::FFT1DAbstract::operator==(const bob::sp::FFT1DAbstract& b) const
{
  return (this->m_length == b.m_length);
}

bool bob::sp::FFT1DAbstract::operator!=(const bob::sp::FFT1DAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::FFT1DAbstract::operator()(const blitz::Array<std::complex<double>,1>& src,
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

void bob::sp::FFT1DAbstract::setLength(const size_t length)
{
  if (length < 1) 
    throw std::runtime_error("FFT length should be at least 1.");
  m_length = length;
  m_wsave.resize(4*length+15);
  initWorkingArray();
  m_buffer.resize(2*length);
}

void bob::sp::FFT1DAbstract::initWorkingArray()
{
  double *wsave_ptr = m_wsave.data();
  cffti((int)m_length, wsave_ptr);
}


bob::sp::FFT1D::FFT1D():
  bob::sp::FFT1DAbstract(1)
{
}

bob::sp::FFT1D::FFT1D(const size_t length):
  bob::sp::FFT1DAbstract(length)
{
}

bob::sp::FFT1D::FFT1D(const bob::sp::FFT1D& other):
  bob::sp::FFT1DAbstract(other)
{
}

bob::sp::FFT1D::~FFT1D()
{
}

bob::sp::FFT1D&
bob::sp::FFT1D::operator=(const FFT1D& other)
{
  if (this != &other) {
    bob::sp::FFT1DAbstract::operator=(other);
  }
  return *this;
}

void bob::sp::FFT1D::setLength(const size_t length)
{
  bob::sp::FFT1DAbstract::setLength(length);
}

void bob::sp::FFT1D::processNoCheck(const blitz::Array<std::complex<double>,1>& src,
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


bob::sp::IFFT1D::IFFT1D():
  bob::sp::FFT1DAbstract(1)
{
}

bob::sp::IFFT1D::IFFT1D(const size_t length):
  bob::sp::FFT1DAbstract(length)
{
}

bob::sp::IFFT1D::IFFT1D(const bob::sp::IFFT1D& other):
  bob::sp::FFT1DAbstract(other)
{
}

bob::sp::IFFT1D::~IFFT1D()
{
}

bob::sp::IFFT1D&
bob::sp::IFFT1D::operator=(const IFFT1D& other)
{
  if (this != &other) {
    bob::sp::FFT1DAbstract::operator=(other);
  }
  return *this;
}


void bob::sp::IFFT1D::setLength(const size_t length)
{
  bob::sp::FFT1DAbstract::setLength(length);
}

void bob::sp::IFFT1D::processNoCheck(const blitz::Array<std::complex<double>,1>& src,
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
