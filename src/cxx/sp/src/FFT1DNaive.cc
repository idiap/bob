/**
 * @file src/cxx/sp/src/FFT1DNaive.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a naive 1D Fast Fourier Transform
 */

#include "sp/FFT1DNaive.h"
#include "core/array_assert.h"

namespace tca = Torch::core::array;
namespace spd = Torch::sp::detail;

spd::FFT1DNaiveAbstract::FFT1DNaiveAbstract( const int length):
  m_length(length), m_wsave(0)
{
  // Initialize working array and normalization factors
  reset();
}

void spd::FFT1DNaiveAbstract::reset(const int length)
{
  // Reset if required
  if( m_length != length) {
    // Update the length
    m_length = length;
    // Reset given the new height and width
    reset();
  }
}
 
void spd::FFT1DNaiveAbstract::reset()
{
  // Precompute working array to save computation time
  initWorkingArray();
}

spd::FFT1DNaiveAbstract::~FFT1DNaiveAbstract()
{
}

void spd::FFT1DNaiveAbstract::initWorkingArray() 
{
  m_wsave.resize(m_length);
  std::complex<double> J(0.,1.);
  for( int i=0; i<m_length; ++i)
    m_wsave(i) = exp(-(J * 2. * (M_PI * i))/(double)m_length);
}

spd::FFT1DNaive::FFT1DNaive( const int length):
  spd::FFT1DNaiveAbstract::FFT1DNaiveAbstract(length)
{
}

void spd::FFT1DNaive::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Check input, inclusive dimension
  tca::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  tca::assertSameShape(src, shape);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::FFT1DNaive::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Compute the FFT
  dst = 0.;
  int ind; // index in the working array using the periodicity of exp(J*x)
  for( int k=0; k<m_length; ++k) {
    for( int n=0; n<m_length; ++n) {
      ind = (n*k) % m_length;
      dst(k) += src(n) * m_wsave(ind);
    }
  }
}


spd::IFFT1DNaive::IFFT1DNaive( const int length):
  spd::FFT1DNaiveAbstract::FFT1DNaiveAbstract(length)
{
}

void spd::IFFT1DNaive::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Check input, inclusive dimension
  tca::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  tca::assertSameShape(src, shape);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::IFFT1DNaive::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Compute the IFFT
  dst = 0.;
  int ind;
  for( int k=0; k<m_length; ++k) {
    for( int n=0; n<m_length; ++n) {
      ind = (((-n*k) % m_length ) + m_length) % m_length;
      dst(k) += src(n) * m_wsave(ind);
    }
  }
  dst /= (double)m_length;
}
