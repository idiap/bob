/**
 * @file cxx/sp/src/FFT1D.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Fast Fourier Transform using FFTPACK
 * functions
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "sp/FFT1D.h"
#include "core/array_assert.h"
#include <fftw3.h>


namespace sp = bob::sp;

bob::sp::FFT1DAbstract::FFT1DAbstract( const size_t length):
  m_length(length)
{
}

bob::sp::FFT1DAbstract::FFT1DAbstract( const bob::sp::FFT1DAbstract& other):
  m_length(other.m_length)
{
}

const bob::sp::FFT1DAbstract& bob::sp::FFT1DAbstract::operator=(const FFT1DAbstract& other)
{
  if(this != &other)
  {
    reset(other.m_length);
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

void bob::sp::FFT1DAbstract::reset(const size_t length)
{
  // Update the length
  m_length = length;
}

void bob::sp::FFT1DAbstract::setLength(const size_t length)
{
  reset(length);
}


bob::sp::FFT1D::FFT1D( const size_t length):
  bob::sp::FFT1DAbstract(length)
{
}

bob::sp::FFT1D::FFT1D( const bob::sp::FFT1D& other):
  bob::sp::FFT1DAbstract(other)
{
}

const bob::sp::FFT1D& bob::sp::FFT1D::operator=(const FFT1D& other)
{
  if(this != &other)
  {
    bob::sp::FFT1DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::FFT1D::operator==(const bob::sp::FFT1D& b) const
{
  return (bob::sp::FFT1DAbstract::operator==(b));
}

bool bob::sp::FFT1D::operator!=(const bob::sp::FFT1D& b) const
{
  return !(this->operator==(b));
}

void bob::sp::FFT1D::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // check input
  bob::core::array::assertCZeroBaseContiguous(src);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Reinterpret cast to fftw format
  fftw_complex* src_ = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>* >(src.data()));
  fftw_complex* dst_ = reinterpret_cast<fftw_complex*>(dst.data());
  
  fftw_plan p;
  // FFTW_ESTIMATE -> The planner is computed quickly but may not be optimized 
  // for large arrays
  p = fftw_plan_dft_1d(src.extent(0), src_, dst_, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}


bob::sp::IFFT1D::IFFT1D( const size_t length):
  bob::sp::FFT1DAbstract(length)
{
}

bob::sp::IFFT1D::IFFT1D( const bob::sp::IFFT1D& other):
  bob::sp::FFT1DAbstract(other)
{
}

const bob::sp::IFFT1D& bob::sp::IFFT1D::operator=(const IFFT1D& other)
{
  if(this != &other)
  {
    bob::sp::FFT1DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::IFFT1D::operator==(const bob::sp::IFFT1D& b) const
{
  return (bob::sp::FFT1DAbstract::operator==(b));
}

bool bob::sp::IFFT1D::operator!=(const bob::sp::IFFT1D& b) const
{
  return !(this->operator==(b));
}

void bob::sp::IFFT1D::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // check input
  bob::core::array::assertCZeroBaseContiguous(src);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Reinterpret cast to fftw format
  fftw_complex* src_ = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>* >(src.data()));
  fftw_complex* dst_ = reinterpret_cast<fftw_complex*>(dst.data());
  
  fftw_plan p;
  // FFTW_ESTIMATE -> The planner is computed quickly but may not be optimized 
  // for large arrays
  p = fftw_plan_dft_1d(src.extent(0), src_, dst_, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p); /* repeat as needed */
  fftw_destroy_plan(p);

  // Rescale as FFTW is not doing it
  dst /= static_cast<double>(m_length);
}

