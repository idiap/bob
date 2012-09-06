/**
 * @file sp/cxx/FFT2D.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK
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

#include "bob/sp/FFT2D.h"
#include "bob/core/array_assert.h"
#include <fftw3.h>


bob::sp::FFT2DAbstract::FFT2DAbstract( const size_t height, const size_t width):
  m_height(height), m_width(width)
{
}

bob::sp::FFT2DAbstract::FFT2DAbstract( const bob::sp::FFT2DAbstract& other):
  m_height(other.m_height), m_width(other.m_width)
{
}

bob::sp::FFT2DAbstract::~FFT2DAbstract()
{
}

const bob::sp::FFT2DAbstract& bob::sp::FFT2DAbstract::operator=(const FFT2DAbstract& other)
{
  if(this != &other)
  {
    reset(other.m_height, other.m_width);
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

void bob::sp::FFT2DAbstract::reset(const size_t height, const size_t width)
{
  // Update the height and width
  m_height = height;
  m_width = width;
}


bob::sp::FFT2D::FFT2D( const size_t height, const size_t width):
  bob::sp::FFT2DAbstract(height, width)
{
}

bob::sp::FFT2D::FFT2D( const bob::sp::FFT2D& other):
  bob::sp::FFT2DAbstract(other)
{
}

bob::sp::FFT2D::~FFT2D()
{
}

const bob::sp::FFT2D& bob::sp::FFT2D::operator=(const FFT2D& other)
{
  if(this != &other)
  {
    bob::sp::FFT2DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::FFT2D::operator==(const bob::sp::FFT2D& b) const
{
  return (bob::sp::FFT2DAbstract::operator==(b));
}

bool bob::sp::FFT2D::operator!=(const bob::sp::FFT2D& b) const
{
  return !(this->operator==(b));
}

void bob::sp::FFT2D::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
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
  p = fftw_plan_dft_2d(src.extent(0), src.extent(1), src_, dst_, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}


void bob::sp::FFT2D::operator()(blitz::Array<std::complex<double>,2>& src_dst)
{
  // check data
  bob::core::array::assertCZeroBaseContiguous(src_dst);

  // Reinterpret cast to fftw format
  fftw_complex* src_dst_ = reinterpret_cast<fftw_complex*>(src_dst.data());

  fftw_plan p;
  // FFTW_ESTIMATE -> The planner is computed quickly but may not be optimized
  // for large arrays
  p = fftw_plan_dft_2d(src_dst.extent(0), src_dst.extent(1), src_dst_, src_dst_, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}


bob::sp::IFFT2D::IFFT2D( const size_t height, const size_t width):
  bob::sp::FFT2DAbstract(height, width)
{
}

bob::sp::IFFT2D::IFFT2D( const bob::sp::IFFT2D& other):
  bob::sp::FFT2DAbstract(other)
{
}

bob::sp::IFFT2D::~IFFT2D()
{
}

const bob::sp::IFFT2D& bob::sp::IFFT2D::operator=(const IFFT2D& other)
{
  if(this != &other)
  {
    bob::sp::FFT2DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::IFFT2D::operator==(const bob::sp::IFFT2D& b) const
{
  return (bob::sp::FFT2DAbstract::operator==(b));
}

bool bob::sp::IFFT2D::operator!=(const bob::sp::IFFT2D& b) const
{
  return !(this->operator==(b));
}

void bob::sp::IFFT2D::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
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
  p = fftw_plan_dft_2d(src.extent(0), src.extent(1), src_, dst_, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Rescale the result by the size of the input 
  // (as this is not performed by FFTW)
  dst /= static_cast<double>(m_width*m_height);
}

void bob::sp::IFFT2D::operator()(blitz::Array<std::complex<double>,2>& src_dst)
{
  // check data
  bob::core::array::assertCZeroBaseContiguous(src_dst);

  // Reinterpret cast to fftw format
  fftw_complex* src_dst_ = reinterpret_cast<fftw_complex*>(src_dst.data());

  fftw_plan p;
  // FFTW_ESTIMATE -> The planner is computed quickly but may not be optimized
  // for large arrays
  p = fftw_plan_dft_2d(src_dst.extent(0), src_dst.extent(1), src_dst_, src_dst_, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Rescale the result by the size of the input
  // (as this is not performed by FFTW)
  src_dst /= static_cast<double>(m_width*m_height);
}
