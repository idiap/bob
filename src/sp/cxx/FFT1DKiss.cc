/**
 * @file sp/cxx/FFT1DKiss.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Fast Fourier Transform
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/sp/FFT1DKiss.h>
#include <bob/core/assert.h>

bob::sp::FFT1DKissAbstract::FFT1DKissAbstract(const size_t length):
  m_length(length)
{
}

bob::sp::FFT1DKissAbstract::FFT1DKissAbstract(
    const bob::sp::FFT1DKissAbstract& other):
  m_length(other.m_length)
{
}

bob::sp::FFT1DKissAbstract::~FFT1DKissAbstract()
{
}

bob::sp::FFT1DKissAbstract& 
bob::sp::FFT1DKissAbstract::operator=(const FFT1DKissAbstract& other)
{
  if (this != &other) {
    m_length = other.m_length;
  }
  return *this;
}

bool bob::sp::FFT1DKissAbstract::operator==(const bob::sp::FFT1DKissAbstract& b) const
{
  return (this->m_length == b.m_length);
}

bool bob::sp::FFT1DKissAbstract::operator!=(const bob::sp::FFT1DKissAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::FFT1DKissAbstract::operator()(const blitz::Array<std::complex<double>,1>& src, 
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

void bob::sp::FFT1DKissAbstract::setLength(const size_t length)
{
  m_length = length;
}
  

bob::sp::FFT1DKiss::FFT1DKiss(const size_t length):
  bob::sp::FFT1DKissAbstract(length),
  m_kissfft(new kissfft<double>(length, false))
{
}

bob::sp::FFT1DKiss::FFT1DKiss(const bob::sp::FFT1DKiss& other):
  bob::sp::FFT1DKissAbstract(other),
  m_kissfft(new kissfft<double>(other.m_length, false))
{
}

bob::sp::FFT1DKiss::~FFT1DKiss()
{
}

bob::sp::FFT1DKiss& 
bob::sp::FFT1DKiss::operator=(const FFT1DKiss& other)
{
  if (this != &other) {
    bob::sp::FFT1DKissAbstract::operator=(other);
    m_kissfft.reset(new kissfft<double>(other.m_length, false));
  }
  return *this;
}

void bob::sp::FFT1DKiss::setLength(const size_t length)
{
  bob::sp::FFT1DKissAbstract::setLength(length);
  m_kissfft.reset(new kissfft<double>(length, false));
}
  
void bob::sp::FFT1DKiss::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst) const
{
  // Compute the FFT
  m_kissfft->transform(src.data(), dst.data());
}


bob::sp::IFFT1DKiss::IFFT1DKiss(const size_t length):
  bob::sp::FFT1DKissAbstract(length),
  m_kissfft(new kissfft<double>(length, true))
{
}

bob::sp::IFFT1DKiss::IFFT1DKiss(const bob::sp::IFFT1DKiss& other):
  bob::sp::FFT1DKissAbstract(other),
  m_kissfft(new kissfft<double>(other.m_length, true))
{
}

bob::sp::IFFT1DKiss::~IFFT1DKiss()
{
}

bob::sp::IFFT1DKiss& 
bob::sp::IFFT1DKiss::operator=(const IFFT1DKiss& other)
{
  if (this != &other) {
    bob::sp::FFT1DKissAbstract::operator=(other);
    m_kissfft.reset(new kissfft<double>(other.m_length, true));
  }
  return *this;
}


void bob::sp::IFFT1DKiss::setLength(const size_t length)
{
  bob::sp::FFT1DKissAbstract::setLength(length);
  m_kissfft.reset(new kissfft<double>(length, true));
}
  
void bob::sp::IFFT1DKiss::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst) const
{
  m_kissfft->transform(src.data(), dst.data());
  dst /= (double)m_length;
}
