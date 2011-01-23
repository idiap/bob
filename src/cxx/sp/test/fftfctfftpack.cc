/**
 * @file src/cxx/sp/test/fftfctfftpack.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Compare FFT and FCT based on FFTPACK with naive DFT DCT 
 * implementations
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sp-DCT_DFT-naive Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "core/Tensor.h"
#include "sp/spDCT.h"
#include "sp/spDFT.h"
#include "sp/spFCT.h"
#include "sp/spFFT.h"

// Random number
#include <cstdlib>

struct T {
  float eps;
  T(): eps(1e-3) { }
  ~T() {}
};

void test_fct1D(const int N, const Torch::FloatTensor& t, float eps) 
{
  // process using DCT
  Torch::spDCT *dct = new Torch::spDCT();
  dct->process(t);
  BOOST_REQUIRE_EQUAL(dct->getNOutputs(), 1);

  // process using FCT
  Torch::spFCT *fct = new Torch::spFCT();
  fct->process(t);
  BOOST_REQUIRE_EQUAL(fct->getNOutputs(), 1);

  // get answers and compare them
  const Torch::FloatTensor &dt_dct = 
    ((const Torch::FloatTensor&)dct->getOutput(0));
  const Torch::FloatTensor &dt_fct = 
    ((const Torch::FloatTensor&)fct->getOutput(0));

  for(int i=0; i < N; ++i)
    BOOST_CHECK_SMALL(dt_fct.get(i)-dt_dct.get(i), eps);

  // process using inverse DCT
  Torch::spFCT *ifct = new Torch::spFCT(true);
  ifct->process(dt_fct);
  BOOST_REQUIRE_EQUAL(ifct->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt_fct = 
    ((const Torch::FloatTensor&)ifct->getOutput(0));
  for(int i=0; i < N; ++i)
    BOOST_CHECK_SMALL(idt_fct.get(i)-t.get(i), eps);

  delete dct;
  delete fct;
  delete ifct;
}


void test_fct2D(const int M, const int N, const Torch::FloatTensor& t, 
  float eps)
{
  // process using DCT
  Torch::spDCT *dct = new Torch::spDCT();
  dct->process(t);
  BOOST_REQUIRE_EQUAL(dct->getNOutputs(), 1);

  // process using FCT
  Torch::spFCT *fct = new Torch::spFCT();
  fct->process(t);
  BOOST_REQUIRE_EQUAL(fct->getNOutputs(), 1);

  // get answers and compare them
  const Torch::FloatTensor &dt_dct = 
    ((const Torch::FloatTensor&)dct->getOutput(0));
  const Torch::FloatTensor &dt_fct = 
    ((const Torch::FloatTensor&)fct->getOutput(0));

  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      BOOST_CHECK_SMALL(dt_fct.get(i,j)-dt_dct.get(i,j), eps);

  // process using inverse FCT
  Torch::spFCT *ifct = new Torch::spFCT(true);
  ifct->process(dt_fct);
  BOOST_REQUIRE_EQUAL(ifct->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt_fct = 
    ((const Torch::FloatTensor&)ifct->getOutput(0));
  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      BOOST_CHECK_SMALL(idt_fct.get(i,j)-t.get(i,j), eps);

  delete dct;
  delete fct;
  delete ifct;
}

void test_fft1D(const int N, const Torch::FloatTensor& t, float eps) 
{
  // process using DFT
  Torch::spDFT *dft = new Torch::spDFT();
  dft->process(t);
  BOOST_REQUIRE_EQUAL(dft->getNOutputs(), 1);

  // process using FFT
  Torch::spFFT *fft = new Torch::spFFT();
  fft->process(t);
  BOOST_REQUIRE_EQUAL(fft->getNOutputs(), 1);

  // get answers and compare them
  const Torch::FloatTensor &dt_dft = 
    ((const Torch::FloatTensor&)dft->getOutput(0));
  const Torch::FloatTensor &dt_fft = 
    ((const Torch::FloatTensor&)fft->getOutput(0));

  for(int i=0; i < N; ++i)
    for(int j=0; j < 2; ++j)
      BOOST_CHECK_SMALL(dt_fft.get(i,j)-dt_dft.get(i,j), eps);

  // process using inverse FFT
  Torch::spFFT *ifft = new Torch::spFFT(true);
  ifft->process(dt_fft);
  BOOST_REQUIRE_EQUAL(ifft->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt_fft = 
    ((const Torch::FloatTensor&)ifft->getOutput(0));
  for(int i=0; i < N; ++i)
    BOOST_CHECK_SMALL(idt_fft.get(i)-t.get(i), eps);

  delete dft;
  delete fft;
  delete ifft;
}

void test_fft2D(const int M, const int N, const Torch::FloatTensor& t,
  float eps)
{
  // process using DFT
  Torch::spDFT *dft = new Torch::spDFT();
  dft->process(t);
  BOOST_REQUIRE_EQUAL(dft->getNOutputs(), 1);

  // process using FFT
  Torch::spFFT *fft = new Torch::spFFT();
  fft->process(t);
  BOOST_REQUIRE_EQUAL(fft->getNOutputs(), 1);

  // get answers and compare them
  const Torch::FloatTensor &dt_dft = 
    ((const Torch::FloatTensor&)dft->getOutput(0));
  const Torch::FloatTensor &dt_fft = 
    ((const Torch::FloatTensor&)fft->getOutput(0));

  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      for(int k=0; k < 2; ++k)
        BOOST_CHECK_SMALL(dt_fft.get(i,j,k)-dt_dft.get(i,j,k), eps);

  // process using inverse FFT
  Torch::spFFT *ifft = new Torch::spFFT(true);
  ifft->process(dt_fft);
  BOOST_REQUIRE_EQUAL(ifft->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt_fft = 
    ((const Torch::FloatTensor&)ifft->getOutput(0));
  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      BOOST_CHECK_SMALL(idt_fft.get(i,j)-t.get(i,j), eps);

  delete dft;
  delete fft;
  delete ifft;
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

/*************** FCT Tests *****************/
BOOST_AUTO_TEST_CASE( test_fct1D_1to64_set )
{
  // size of the data
  for(int N=1; N < 65; ++N) {
    // set up simple 1D tensor
    Torch::FloatTensor t(N);
    for(int i=0; i<N; ++i)
      t.set(i, 1.0f+i);

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

    // set up simple 1D random tensor 
    Torch::FloatTensor t(N);
    for( int i=0; i < N; ++i)
      t.set(i,(float) ((rand()/RAND_MAX)*10.f));

    // call the test function
    test_fct1D(N, t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fct2D_1x1to8x8_set )
{
  // size of the data
  for(int M=1; M < 9; ++M)
    for(int N=1; N < 9; ++N) {
      // set up simple 2D tensor
      Torch::FloatTensor t(M,N);
      for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
          t.set(i,j, 1.0f+i+j);

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

    // set up simple 2D random tensor 
    Torch::FloatTensor t(M,N);
    for( int i=0; i < M; ++i)
      for( int j=0; j < N; ++j)
        t.set(i,j, (float) ((rand()/RAND_MAX)*10.f));

    // call the test function
    test_fct2D(M, N, t, eps);
  }
}


/*************** FFT Tests *****************/
BOOST_AUTO_TEST_CASE( test_fft1D_1to64_set )
{
  // size of the data
  for(int N=1; N < 65; ++N) {
    // set up simple 1D tensor
    Torch::FloatTensor t(N);
    for(int i=0; i<N; ++i)
      t.set(i, 1.0f+i);

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
    Torch::FloatTensor t(N);
    for( int i=0; i < N; ++i)
      t.set(i,(float) ((rand()/RAND_MAX)*10.f));

    // call the test function
    test_fft1D(N, t, eps);
  }
}

BOOST_AUTO_TEST_CASE( test_fft2D_1x1to8x8_set )
{
  // size of the data
  for(int M=1; M < 9; ++M)
    for(int N=1; N < 9; ++N) {
      // set up simple 2D tensor
      Torch::FloatTensor t(M,N);
      for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
          t.set(i,j, 1.0f+i+j);

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

    // set up simple 2D random tensor 
    Torch::FloatTensor t(M,N);
    for( int i=0; i < M; ++i)
      for( int j=0; j < N; ++j)
        t.set(i,j, (float) ((rand()/RAND_MAX)*10.f));

    // call the test function
    test_fft2D(M, N, t, eps);
  }
}


BOOST_AUTO_TEST_SUITE_END()

