/**
 * @file cxx/sp/src/FFT1D.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Fast Fourier Transform using FFTPACK
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

#include "sp/FFT1D.h"
#include "core/array_assert.h"
#include <fftw3.h>


namespace tca = bob::core::array;
namespace sp = bob::sp;

sp::FFT1DAbstract::FFT1DAbstract( const int length):
  m_length(length)
{
  // Initialize working array and normalization factors
  reset();
}

void sp::FFT1DAbstract::reset(const int length)
{
  // Reset if required
  if( m_length != length) {
    // Update the length
    m_length = length;
    // Deallocate memory
    cleanup();
    // Reset given the new height and width
    reset();
  }
}
 
void sp::FFT1DAbstract::reset()
{
}

sp::FFT1DAbstract::~FFT1DAbstract()
{
  cleanup();
}

void sp::FFT1DAbstract::cleanup() {
}

sp::FFT1D::FFT1D( const int length):
  sp::FFT1DAbstract::FFT1DAbstract(length)
{
}

void sp::FFT1D::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
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
  p = fftw_plan_dft_1d(src.extent(0), src_, dst_, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}


sp::IFFT1D::IFFT1D( const int length):
  sp::FFT1DAbstract::FFT1DAbstract(length)
{
}

void sp::IFFT1D::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
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
  p = fftw_plan_dft_1d(src.extent(0), src_, dst_, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p); /* repeat as needed */
  fftw_destroy_plan(p);

  // Rescale as FFTW is not doing it
  dst /= static_cast<double>(m_length);
}

