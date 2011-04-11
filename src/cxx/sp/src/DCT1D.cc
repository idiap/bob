/**
 * @file src/cxx/sp/src/DCT1D.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based 1D Fast Cosine Transform using FFTPACK 
 * functions
 */

#include "sp/DCT1D.h"
#include "core/array_assert.h"

// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cosqi_( int *n, double *wsave);
extern "C" void cosqf_( int *n, double *x, double *wsave);
extern "C" void cosqb_( int *n, double *x, double *wsave);


namespace sp = Torch::sp;

sp::DCT1DAbstract::DCT1DAbstract( const int length):
  m_length(length), m_wsave(0)
{
  // Initialize working array and normalization factors
  reset();
}

void sp::DCT1DAbstract::reset(const int length)
{
  // Update the length
  m_length = length;

  // Reset given the new height and width
  reset();
}
 
void sp::DCT1DAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();

  // Precompute working array to save computation time
  initWorkingArray();
}

sp::DCT1DAbstract::~DCT1DAbstract()
{
  cleanup();
}

void sp::DCT1DAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1byl=sqrt(1./m_length);
  m_sqrt_2byl=sqrt(2./m_length);
  m_sqrt_1l=sqrt(1.*m_length);
  m_sqrt_2l=sqrt(2.*m_length);
}

void sp::DCT1DAbstract::initWorkingArray() 
{
  int n_wsave = 3*m_length+15;
  m_wsave = new double[n_wsave];
  cosqi_( &m_length, m_wsave);
}

void sp::DCT1DAbstract::cleanup() {
  if(m_wsave)
    delete [] m_wsave;
}

sp::DCT1D::DCT1D( const int length):
  sp::DCT1DAbstract::DCT1DAbstract(length)
{
}

void sp::DCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // check input
  Torch::core::assertZeroBase(src);

  // Check output
  Torch::core::assertCZeroBaseContiguous(dst);
  Torch::core::assertSameShape( dst, src);

  // Copy content from src to dst
  dst = src;

  // Compute the FCT
  cosqb_( &m_length, dst.data(), m_wsave);

  // Normalize
  dst(0) *= m_sqrt_1byl/4.;
  blitz::Range r_dst(1, dst.ubound(0) );
  dst(r_dst) *= m_sqrt_2byl/4.;
}


sp::IDCT1D::IDCT1D( const int length):
  sp::DCT1DAbstract::DCT1DAbstract(length)
{
}

void sp::IDCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // check input
  Torch::core::assertZeroBase(src);

  // Check output
  Torch::core::assertCZeroBaseContiguous(dst);
  Torch::core::assertSameShape( dst, src);

  // Copy content from src to dst
  dst = src;

  // Normalize
  dst(0) /= m_sqrt_1l;
  blitz::Range r_dst(1, dst.ubound(0) );
  dst(r_dst) /= m_sqrt_2l;

  // Compute the FCT
  cosqf_( &m_length, dst.data(), m_wsave);
}

