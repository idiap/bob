/**
 * @file src/cxx/sp/test/fftfctfftpack_blitz.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Compare FFT and FCT based on FFTPACK with naive DFT DCT 
 * implementations.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sp-FCT_FFT-fftpack-blitz Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "sp/FFT.h"
#include "sp/FCT.h"

// Random number
#include <cstdlib>
#include <iostream>

struct T {
  double eps;
  T(): eps(1e-3) { }
  ~T() {}
};


void test_fct1D(const int N, const blitz::Array<double,1> t, double eps) 
{
  // process using FCT
  blitz::Array<double,1> t_fct = Torch::sp::fct(t);
  BOOST_REQUIRE_EQUAL(t_fct.extent(0), t.extent(0));

  // TODO: get DCT answer and compare with FCT

  // process using inverse FCT
  blitz::Array<double,1> t_fct_ifct = Torch::sp::ifct(t_fct);
  BOOST_REQUIRE_EQUAL(t_fct_ifct.extent(0), t.extent(0));

  // Compare to original
  for(int i=0; i < t.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs(t_fct_ifct(i)-t(i)), eps);
}


void test_fct2D(const int M, const int N, 
  const blitz::Array<double,2> t, double eps) 
{
  // process using FCT
  blitz::Array<double,2> t_fct = Torch::sp::fct(t);
  BOOST_REQUIRE_EQUAL(t_fct.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fct.extent(1), t.extent(1));

  // TODO: get DCT answer and compare with FCT

  // process using inverse FCT
  blitz::Array<double,2> t_fct_ifct = Torch::sp::ifct(t_fct);
  BOOST_REQUIRE_EQUAL(t_fct_ifct.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fct_ifct.extent(1), t.extent(1));

  // Compare to original
  for(int i=0; i < t.extent(0); ++i)
    for(int j=0; j < t.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t_fct_ifct(i,j)-t(i,j)), eps);
}

void test_fft1D(const int N, const blitz::Array<std::complex<double>,1> t, double eps) 
{
  // process using FFT
  blitz::Array<std::complex<double>,1> t_fft = Torch::sp::fft(t);
  BOOST_REQUIRE_EQUAL(t_fft.extent(0), t.extent(0));

  // TODO: get DFT answer and compare with FFT

  // process using inverse FFT
  blitz::Array<std::complex<double>,1> t_fft_ifft = Torch::sp::ifft(t_fft);
  BOOST_REQUIRE_EQUAL(t_fft_ifft.extent(0), t.extent(0));

  // Compare to original
  for(int i=0; i < t.extent(0); ++i)
    BOOST_CHECK_SMALL( abs(t_fft_ifft(i)-t(i)), eps);
}


void test_fft2D(const int M, const int N, 
  const blitz::Array<std::complex<double>,2> t, double eps) 
{
  // process using FFT
  blitz::Array<std::complex<double>,2> t_fft = Torch::sp::fft(t);
  BOOST_REQUIRE_EQUAL(t_fft.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fft.extent(1), t.extent(1));

  // TODO: get DFT answer and compare with FFT

  // process using inverse FFT
  blitz::Array<std::complex<double>,2> t_fft_ifft = Torch::sp::ifft(t_fft);
  BOOST_REQUIRE_EQUAL(t_fft_ifft.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fft_ifft.extent(1), t.extent(1));

  // Compare to original
  for(int i=0; i < t.extent(0); ++i)
    for(int j=0; j < t.extent(1); ++j)
      BOOST_CHECK_SMALL( abs(t_fft_ifft(i,j)-t(i,j)), eps);
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

/*************** FCT Tests *****************/
BOOST_AUTO_TEST_CASE( test_fct1D_1to64_set )
{
  // size of the data
  for(int N=1; N <65; ++N) {
    // set up simple 1D array
    blitz::Array<double,1> t(N);
    for(int i=0; i<N; ++i)
      t(i) = 1.0+i;

    // call the test function
    test_fct1D(N, t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fct1D_range1to2048_random )
{
  // This tests the 1D FCT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for(int loop=0; loop < 10; ++loop) {
    // size of the data
    int N = (rand() % 2048 + 1);//random.randint(1,2048)

    // set up simple 1D random array
    blitz::Array<double,1> t(N);
    for(int i=0; i<N; ++i)
      t(i) = (rand()/RAND_MAX)*10.;

    // call the test function
    test_fct1D(N, t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fct2D_1x1to8x8_set )
{
  // size of the data
  for(int M=1; M < 9; ++M)
    for(int N=1; N < 9; ++N) {
      // set up simple 1D array
      blitz::Array<double,2> t(M,N);
      for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
          t(i,j) = 1.0+i+j;

      // call the test function
      test_fct2D(M, N, t, eps);
    }
}

BOOST_AUTO_TEST_CASE( test_fct2D_range1x1to64x64_random )
{
  // This tests the 1D FCT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for(int loop=0; loop < 10; ++loop) {
    // size of the data
    int M = (rand() % 64 + 1);
    int N = (rand() % 64 + 1);

    // set up simple 1D random array
    blitz::Array<double,2> t(M,N);
    for( int i=0; i < M; ++i)
      for( int j=0; j < N; ++j)
        t(i,j) = (rand()/RAND_MAX)*10.;

    // call the test function
    test_fct2D(M, N, t, eps);
  }
}


/*************** FFT Tests *****************/
BOOST_AUTO_TEST_CASE( test_fft1D_1to64_set )
{
  // size of the data
  for(int N=1; N <65 ; ++N) {
    // set up simple 1D tensor
    blitz::Array<std::complex<double>,1> t(N);
    for(int i=0; i<N; ++i)
      t(i) = std::complex<double>(1.0+i,0);

    // call the test function
    test_fft1D(N, t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fft1D_range1to2048_random )
{
  // This tests the 1D FFT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for(int loop=0; loop < 10; ++loop) {
    // size of the data
    int N = (rand() % 2048 + 1);//random.randint(1,2048)

    // set up simple 1D random tensor 
    blitz::Array<std::complex<double>,1> t(N);
    for(int i=0; i<N; ++i)
      t(i) = (rand()/RAND_MAX)*10.;

    // call the test function
    test_fft1D(N, t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fft2D_1x1to8x8_set )
{
  // size of the data
  for(int M=1; M < 9; ++M)
    for(int N=1; N < 9; ++N) {
      // set up simple 1D tensor
      blitz::Array<std::complex<double>,2> t(M,N);
      for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
          t(i,j) = std::complex<double>(1.0+i+j,0);

      // call the test function
      test_fft2D(M, N, t, eps);
    }
}

BOOST_AUTO_TEST_CASE( test_fft2D_range1x1to64x64_random )
{
  // This tests the 1D FCT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for(int loop=0; loop < 10; ++loop) {
    // size of the data
    int M = (rand() % 64 + 1);
    int N = (rand() % 64 + 1);

    // set up simple 1D random tensor 
    blitz::Array<std::complex<double>,2> t(M,N);
    for( int i=0; i < M; ++i)
      for( int j=0; j < N; ++j)
        t(i,j) = std::complex<double>((rand()/RAND_MAX)*10.,0);

    // call the test function
    test_fft2D(M, N, t, eps);
  }
}


BOOST_AUTO_TEST_SUITE_END()

