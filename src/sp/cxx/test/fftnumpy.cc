/**
 * @file sp/cxx/test/fftnumpy.cc
 * @date Fri Nov 15 09:57:58 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Compare FFT and FCT based on NumPy with naive DFT DCT
 * implementations.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sp-FCT_FFT-NUMPY Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <bob/sp/fftshift.h>
#include <bob/sp/FFT1DNumpy.h>
#include <bob/sp/FFT1DNaive.h>
#include <bob/sp/FFT2DNumpy.h>
#include <bob/sp/FFT2DNaive.h>
#include <bob/sp/DCT1DNumpy.h>
#include <bob/sp/DCT1DNaive.h>
#include <bob/sp/DCT2DNumpy.h>
#include <bob/sp/DCT2DNaive.h>

// Random number
#include <cstdlib>

struct T {
  double eps;
  T(): eps(1e-3) { }
  ~T() {}
};


void test_fct1D( const blitz::Array<double,1> t, double eps)
{
  // process using FCT
  blitz::Array<double,1> t_fct(t.extent(0)), t_dct(t.extent(0));
  bob::sp::DCT1DNumpy fct(t.extent(0));
  fct(t, t_fct);
  BOOST_REQUIRE_EQUAL(t_fct.extent(0), t.extent(0));

  // get DCT answer and compare with FCT
  bob::sp::detail::DCT1DNaive dct_new_naive(t.extent(0));
  dct_new_naive(t, t_dct);
  // Compare
  for (int i=0; i < t_fct.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs(t_fct(i)-t_dct(i)), eps);

  // process using inverse FCT
  blitz::Array<double,1> t_fct_ifct(t.extent(0));
  bob::sp::IDCT1DNumpy ifct(t.extent(0));
  ifct(t_fct, t_fct_ifct);
  BOOST_REQUIRE_EQUAL(t_fct_ifct.extent(0), t.extent(0));

  // Compare to original
  for (int i=0; i < t.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs(t_fct_ifct(i)-t(i)), eps);

  // process using inverse DCT
  blitz::Array<double,1> t_dct_idct(t.extent(0));
  bob::sp::detail::IDCT1DNaive idct_naive(t.extent(0));
  ifct(t_dct, t_dct_idct);
  BOOST_REQUIRE_EQUAL(t_dct_idct.extent(0), t.extent(0));

  // Compare to original
  for (int i=0; i < t.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs(t_dct_idct(i)-t(i)), eps);
}

void test_fct2D( const blitz::Array<double,2> t, double eps)
{
  // process using FCT
  blitz::Array<double,2> t_fct(t.extent(0), t.extent(1)),
    t_dct(t.extent(0), t.extent(1));
  bob::sp::DCT2DNumpy fct(t.extent(0), t.extent(1));
  fct(t, t_fct);
  BOOST_REQUIRE_EQUAL(t_fct.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fct.extent(1), t.extent(1));

  // get DCT answer and compare with FCT
  bob::sp::detail::DCT2DNaive dct_new_naive(t.extent(0), t.extent(1));
  dct_new_naive(t, t_dct);
  // Compare
  for (int i=0; i < t_fct.extent(0); ++i)
    for (int j=0; j < t_fct.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t_fct(i,j)-t_dct(i,j)), eps);

  // process using inverse FCT
  blitz::Array<double,2> t_fct_ifct(t.extent(0), t.extent(1));
  bob::sp::IDCT2DNumpy ifct(t.extent(0), t.extent(1));
  ifct(t_fct, t_fct_ifct);
  BOOST_REQUIRE_EQUAL(t_fct_ifct.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fct_ifct.extent(1), t.extent(1));

  // Compare to original
  for (int i=0; i < t.extent(0); ++i)
    for (int j=0; j < t.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t_fct_ifct(i,j)-t(i,j)), eps);

  // process using inverse DCT
  blitz::Array<double,2> t_dct_idct(t.extent(0), t.extent(1));
  bob::sp::detail::IDCT2DNaive idct_naive(t.extent(0), t.extent(1));
  ifct(t_dct, t_dct_idct);
  BOOST_REQUIRE_EQUAL(t_dct_idct.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_dct_idct.extent(1), t.extent(1));

  // Compare to original
  for (int i=0; i < t.extent(0); ++i)
    for (int j=0; j < t.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t_dct_idct(i,j)-t(i,j)), eps);
}


void test_fft1D( const blitz::Array<std::complex<double>,1> t, double eps)
{
  // process using FFT
  blitz::Array<std::complex<double>,1> t_fft(t.extent(0)), t_dft(t.extent(0));
  bob::sp::FFT1DNumpy fft(t.extent(0));
  fft(t, t_fft);
  BOOST_REQUIRE_EQUAL(t_fft.extent(0), t.extent(0));

  // get DFT answer and compare with FFT
  bob::sp::detail::FFT1DNaive dft_new_naive(t.extent(0));
  dft_new_naive(t, t_dft);
  // Compare
  for (int i=0; i < t_fft.extent(0); ++i)
    BOOST_CHECK_SMALL( abs(t_fft(i)-t_dft(i)), eps);

  // process using inverse FFT
  blitz::Array<std::complex<double>,1> t_fft_ifft(t.extent(0));
  bob::sp::IFFT1DNumpy ifft(t.extent(0));
  ifft(t_fft, t_fft_ifft);
  BOOST_REQUIRE_EQUAL(t_fft_ifft.extent(0), t.extent(0));

  // Compare to original
  for (int i=0; i < t.extent(0); ++i)
    BOOST_CHECK_SMALL( abs(t_fft_ifft(i)-t(i)), eps);
}

void test_fft2D( const blitz::Array<std::complex<double>,2> t, double eps)
{
  // process using FFT
  blitz::Array<std::complex<double>,2> t_fft(t.extent(0), t.extent(1)),
    t_dft(t.extent(0), t.extent(1));
  bob::sp::FFT2DNumpy fft(t.extent(0), t.extent(1));
  fft(t, t_fft);
  BOOST_REQUIRE_EQUAL(t_fft.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fft.extent(1), t.extent(1));

  // get DFT answer and compare with FFT
  bob::sp::detail::FFT2DNaive dft_new_naive(t.extent(0), t.extent(1));
  dft_new_naive(t, t_dft);
  // Compare
  for (int i=0; i < t_fft.extent(0); ++i)
    for (int j=0; j < t_fft.extent(1); ++j)
      BOOST_CHECK_SMALL( abs(t_fft(i,j)-t_dft(i,j)), eps);

  // process using inverse FFT
  blitz::Array<std::complex<double>,2> t_fft_ifft(t.extent(0), t.extent(1));
  bob::sp::IFFT2DNumpy ifft(t.extent(0), t.extent(1));
  ifft(t_fft, t_fft_ifft);
  BOOST_REQUIRE_EQUAL(t_fft_ifft.extent(0), t.extent(0));
  BOOST_REQUIRE_EQUAL(t_fft_ifft.extent(1), t.extent(1));

  // Compare to original
  for (int i=0; i < t.extent(0); ++i)
    for (int j=0; j < t.extent(1); ++j)
      BOOST_CHECK_SMALL( abs(t_fft_ifft(i,j)-t(i,j)), eps);
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

/*************** FCT Tests *****************/
BOOST_AUTO_TEST_CASE( test_fct1D_1to64_set )
{
  // size of the data
  for (int N=1; N <65; ++N) {
    // set up simple 1D array
    blitz::Array<double,1> t(N);
    for (int i=0; i<N; ++i)
      t(i) = 1.0+i;

    // call the test function
    test_fct1D( t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fct1D_range1to2048_random )
{
  // This tests the 1D FCT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for (int loop=0; loop < 10; ++loop) {
    // size of the data
    int N = (rand() % 2048 + 1);//random.randint(1,2048)

    // set up simple 1D random array
    blitz::Array<double,1> t(N);
    for (int i=0; i<N; ++i)
      t(i) = (rand()/(double)RAND_MAX)*10.;

    // call the test function
    test_fct1D( t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fct2D_1x1to8x8_set )
{
  // size of the data
  for (int M=1; M < 9; ++M)
    for (int N=1; N < 9; ++N) {
      // set up simple 1D array
      blitz::Array<double,2> t(M,N);
      for (int i=0; i<M; ++i)
        for (int j=0; j<N; ++j)
          t(i,j) = 1.0+i+j;

      // call the test function
      test_fct2D( t, eps);
    }
}

BOOST_AUTO_TEST_CASE( test_fct2D_range1x1to64x64_random )
{
  // This tests the 1D FCT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for (int loop=0; loop < 10; ++loop) {
    // size of the data
    int M = (rand() % 64 + 1);
    int N = (rand() % 64 + 1);

    // set up simple 1D random array
    blitz::Array<double,2> t(M,N);
    for (int i=0; i < M; ++i)
      for (int j=0; j < N; ++j)
        t(i,j) = (rand()/(double)RAND_MAX)*10.;

    // call the test function
    test_fct2D( t, eps);
  }
}


/*************** FFT Tests *****************/
BOOST_AUTO_TEST_CASE( test_fft1D_1to64_set )
{
  // size of the data
  for (int N=1; N <65 ; ++N) {
    // set up simple 1D tensor
    blitz::Array<std::complex<double>,1> t(N);
    for (int i=0; i<N; ++i)
      t(i) = std::complex<double>(1.0+i,0);

    // call the test function
    test_fft1D( t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fft1D_range1to2048_random )
{
  // This tests the 1D FFT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for (int loop=0; loop < 10; ++loop) {
    // size of the data
    int N = (rand() % 2048 + 1);//random.randint(1,2048)

    // set up simple 1D random tensor
    blitz::Array<std::complex<double>,1> t(N);
    for (int i=0; i<N; ++i)
      t(i) = (rand()/(double)RAND_MAX)*10.;

    // call the test function
    test_fft1D( t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fft2D_1x1to8x8_set )
{
  // size of the data
  for (int M=1; M < 9; ++M)
    for (int N=1; N < 9; ++N) {
      // set up simple 1D tensor
      blitz::Array<std::complex<double>,2> t(M,N);
      for (int i=0; i<M; ++i)
        for (int j=0; j<N; ++j)
          t(i,j) = std::complex<double>(1.0+i+j,0);

      // call the test function
      test_fft2D( t, eps);
    }
}

BOOST_AUTO_TEST_CASE( test_fft2D_range1x1to64x64_random )
{
  // This tests the 2D FCT using 10 random vectors
  // The size of each vector is randomly chosen between 3 and 2048
  for (int loop=0; loop < 10; ++loop) {
    // size of the data
    int M = (rand() % 64 + 1);
    int N = (rand() % 64 + 1);

    // set up simple 1D random tensor
    blitz::Array<std::complex<double>,2> t(M,N);
    for (int i=0; i < M; ++i)
      for (int j=0; j < N; ++j)
        t(i,j) = std::complex<double>((rand()/(double)RAND_MAX)*10.,0);

    // call the test function
    test_fft2D( t, eps);
  }
}

BOOST_AUTO_TEST_SUITE_END()
