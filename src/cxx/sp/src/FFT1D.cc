/**
 * @file src/cxx/sp/src/FFT1D.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based 1D Fast Fourier Transform using FFTPACK 
 * functions
 */

#include "sp/FFT1D.h"
#include "core/array_assert.h"

// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cffti_( int *n, double *wsave);
extern "C" void cfftf_( int *n, std::complex<double> *x, double *wsave);
extern "C" void cfftb_( int *n, std::complex<double> *x, double *wsave);

namespace tca = Torch::core::array;
namespace sp = Torch::sp;

sp::FFT1DAbstract::FFT1DAbstract( const int length):
  m_length(length), m_wsave(0)
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
  // Precompute working array to save computation time
  initWorkingArray();
}

sp::FFT1DAbstract::~FFT1DAbstract()
{
  cleanup();
}

void sp::FFT1DAbstract::initWorkingArray() 
{
  int n_wsave = 4*m_length+15;
  m_wsave = new double[n_wsave];
  cffti_( &m_length, m_wsave);
}

void sp::FFT1DAbstract::cleanup() {
  if(m_wsave)
    delete [] m_wsave;
}

sp::FFT1D::FFT1D( const int length):
  sp::FFT1DAbstract::FFT1DAbstract(length)
{
}

void sp::FFT1D::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Copy content from src to dst
  dst = src;

  // Compute the FFT
  cfftf_( &m_length, dst.data(), m_wsave);
}


sp::IFFT1D::IFFT1D( const int length):
  sp::FFT1DAbstract::FFT1DAbstract(length)
{
}

void sp::IFFT1D::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Copy content from src to dst
  dst = src;

  // Compute the FFT
  cfftb_( &m_length, dst.data(), m_wsave);

  dst /= static_cast<double>(m_length);
}

