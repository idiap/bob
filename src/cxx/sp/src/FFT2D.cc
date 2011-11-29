/**
 * @file cxx/sp/src/FFT2D.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK
 * functions
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "sp/FFT2D.h"
#include "core/array_assert.h"
#include <fftw3.h>


namespace tca = Torch::core::array;
namespace sp = Torch::sp;

sp::FFT2DAbstract::FFT2DAbstract( const int height, const int width):
  m_height(height), m_width(width)
{
  reset();
}

sp::FFT2DAbstract::~FFT2DAbstract()
{
  cleanup();
}

void sp::FFT2DAbstract::reset(const int height, const int width)
{
  // Reset if required
  if( m_height != height || m_width != width) {
    // Update the height and width
    m_height = height;
    m_width = width;
    // Deallocate memory
    cleanup();
    // Reset given the new height and width
    reset();
  }
}
 
void sp::FFT2DAbstract::reset()
{
}

void sp::FFT2DAbstract::cleanup() {
}



sp::FFT2D::FFT2D( const int height, const int width):
  sp::FFT2DAbstract::FFT2DAbstract(height, width)
{
}

void sp::FFT2D::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // check input
  tca::assertCZeroBaseContiguous(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

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


sp::IFFT2D::IFFT2D( const int height, const int width):
  sp::FFT2DAbstract::FFT2DAbstract(height, width)
{
}

void sp::IFFT2D::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // check input
  tca::assertCZeroBaseContiguous(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

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

