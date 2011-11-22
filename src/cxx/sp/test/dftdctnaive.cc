/**
 * @file cxx/sp/test/dftdctnaive.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Compare naive DFT and DCT with some values returned by Matlab
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sp-DCT_DFT-naive Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "core/Tensor.h"
#include "sp/spDCT.h"
#include "sp/spDFT.h"


struct T {
  float eps;
  T(): eps(1e-3) { }
  ~T() {}
};

void test_dct1D(const int N, const Torch::FloatTensor& t, const float mat[],
  float eps) 
{
  // process using DCT
  Torch::spDCT *dct = new Torch::spDCT();
  dct->process(t);
  BOOST_REQUIRE_EQUAL(dct->getNOutputs(), 1);

  // get answer and compare to matlabs dct
  const Torch::FloatTensor &dt = 
    ((const Torch::FloatTensor&)dct->getOutput(0));

  for(int i=0; i < N; ++i)
    BOOST_CHECK_SMALL(dt.get(i)-mat[i], eps);

  // process using inverse DCT
  Torch::spDCT *idct = new Torch::spDCT(true);
  idct->process(dt);
  BOOST_REQUIRE_EQUAL(idct->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt = 
    ((const Torch::FloatTensor&)idct->getOutput(0));
  for(int i=0; i < N; ++i)
    BOOST_CHECK_SMALL(idt.get(i)-t.get(i), eps);

  delete dct;
  delete idct;
}


void test_dct2D(const int M, const int N, const Torch::FloatTensor& t, 
  const float mat[], float eps) 
{
  // process using DCT
  Torch::spDCT *dct = new Torch::spDCT();
  dct->process(t);
  BOOST_REQUIRE_EQUAL(dct->getNOutputs(), 1);

  // get answer and compare to matlabs dct
  const Torch::FloatTensor &dt = 
    ((const Torch::FloatTensor&)dct->getOutput(0));

  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      BOOST_CHECK_SMALL(dt.get(i,j)-mat[i*N+j], eps);

  // process using inverse DCT
  Torch::spDCT *idct = new Torch::spDCT(true);
  idct->process(dt);
  BOOST_REQUIRE_EQUAL(idct->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt = 
    ((const Torch::FloatTensor&)idct->getOutput(0));
  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      BOOST_CHECK_SMALL(idt.get(i,j)-t.get(i,j), eps);

  delete dct;
  delete idct;
}

void test_dft1D(const int N, const Torch::FloatTensor& t, const float mat[],
  float eps) 
{
  // process using DFT
  Torch::spDFT *dft = new Torch::spDFT();
  dft->process(t);
  BOOST_REQUIRE_EQUAL(dft->getNOutputs(), 1);

  // get answer and compare to matlabs dct
  const Torch::FloatTensor &dt = 
    ((const Torch::FloatTensor&)dft->getOutput(0));

  for(int i=0; i < N; ++i)
    for(int j=0; j < 2; ++j)
      BOOST_CHECK_SMALL(dt.get(i,j)-mat[i*2+j], eps);

  // process using inverse DFT
  Torch::spDFT *idft = new Torch::spDFT(true);
  idft->process(dt);
  BOOST_REQUIRE_EQUAL(idft->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt = 
    ((const Torch::FloatTensor&)idft->getOutput(0));
  for(int i=0; i < N; ++i)
    BOOST_CHECK_SMALL(idt.get(i)-t.get(i), eps);

  delete dft;
  delete idft;
}

void test_dft2D(const int M, const int N, const Torch::FloatTensor& t,
  const float mat[], float eps)
{
  // process using DFT
  Torch::spDFT *dft = new Torch::spDFT();
  dft->process(t);
  BOOST_REQUIRE_EQUAL(dft->getNOutputs(), 1);

  // get answer and compare to matlabs dct
  const Torch::FloatTensor &dt = 
    ((const Torch::FloatTensor&)dft->getOutput(0));

  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      for(int k=0; k < 2; ++k)
        BOOST_CHECK_SMALL(dt.get(i,j,k)-mat[i*2*N+j*2+k], eps);

  // process using inverse DFT
  Torch::spDFT *idft = new Torch::spDFT(true);
  idft->process(dt);
  BOOST_REQUIRE_EQUAL(idft->getNOutputs(), 1);

  // get answer and compare to original
  const Torch::FloatTensor &idt = 
    ((const Torch::FloatTensor&)idft->getOutput(0));
  for(int i=0; i < M; ++i)
    for(int j=0; j < N; ++j)
      BOOST_CHECK_SMALL(idt.get(i,j)-t.get(i,j), eps);

  delete dft;
  delete idft;
}



BOOST_FIXTURE_TEST_SUITE( test_setup, T )

//this a DCT1D with a 1D vector of length 3 against Matlab
BOOST_AUTO_TEST_CASE( test_dct1D_3 )
{
  // size of the data
  const int N = 3;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[N] = {3.4641f, -1.4142f, 0.f};

  // call the test function
  test_dct1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dct1D_5 )
{
  // size of the data
  const int N = 5;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[N] = {6.7082f, -3.1495f, 0.f, -0.2840f, 0.f};

  // call the test function
  test_dct1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dct1D_8 )
{
  // size of the data
  const int N = 8;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[N] = {12.7279f, -6.4423f, 0.f, -0.6735f,
    0.f, -0.2009f, 0.f, -0.0507f};

  // call the test function
  test_dct1D(N, t, mat, eps);
}


BOOST_AUTO_TEST_CASE( test_dct1D_17 )
{
  // size of the data
  const int N = 17;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[N] = 
    {37.1080f,-20.0585f,0.f,-2.2025f,0.f,-0.7727f,0.f,-0.3768f,
     0.f,-0.2116f,0.f,-0.1249f,0.f,-0.0713f,0.f,-0.0326f,0.f};

  // call the test function
  test_dct1D(N, t, mat, eps);
}


BOOST_AUTO_TEST_CASE( test_dct2D_2x2a )
{
  // size of the data
  const int M = 2;
  const int N = 2;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  t.set(0, 0, 1.0f);
  t.set(0, 1, 0.0f);
  t.set(1, 0, 0.0f);
  t.set(1, 1, 0.0f);

  // array containing matlab values
  const float mat[M*N] = {0.5f, 0.5f, 0.5f, 0.5f};

  // call the test function
  test_dct2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dct2D_2x2b )
{
  // size of the data
  const int M = 2;
  const int N = 2;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  t.set(0, 0, 3.2f);
  t.set(0, 1, 4.7f);
  t.set(1, 0, 5.4f);
  t.set(1, 1, 0.2f);

  // array containing matlab values
  const float mat[M*N] = {6.75f, 1.85f, 1.15f, -3.35f};

  // call the test function
  test_dct2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dct2D_4x4 )
{
  // size of the data
  const int M = 4;
  const int N = 4;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  for( int i=0; i < M; ++i)
    for( int j=0; j < N; ++j)
      t.set(i, j, 1.0f+i+j);

  // array containing matlab values
  const float mat[M*N] = 
    {16.f, -4.4609f, 0.f, -0.3170f, -4.4609f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, -0.3170f, 0.f, 0.f, 0.f};

  // call the test function
  test_dct2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dct2D_8x8 )
{
  // size of the data
  const int M = 8;
  const int N = 8;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  for( int i=0; i < M; ++i)
    for( int j=0; j < N; ++j)
      t.set(i, j, 1.0f+i+j);

  // array containing matlab values
  const float mat[M*N] = 
    {64.f, -18.2216f, 0.f, -1.9048f, 0.f, -0.5682f, 0.f, -0.1434f,
     -18.2216f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -1.9048f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -0.5682f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -0.1434f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // call the test function
  test_dct2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dct2D_16x16 )
{
  // size of the data
  const int M = 16;
  const int N = 16;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  for( int i=0; i < M; ++i)
    for( int j=0; j < N; ++j)
      t.set(i, j, 1.0f+i+j);

  // array containing matlab values
  const float mat[M*N] = 
    {256.f, -73.2461f, 0.f, -8.0301f, 0.f, -2.8063f, 0.f, -1.3582f,
     0.f, -0.7507f, 0.f, -0.4286f, 0.f, -0.2242f, 0.f, -0.0700f,
     -73.2461f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -8.0301f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -2.8063f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -1.3582f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -0.7507f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -0.4286f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -0.2242f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     -0.0700f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // call the test function
  test_dct2D(M, N, t, mat, eps);
}

/******** DFT *************/
BOOST_AUTO_TEST_CASE( test_dft1D_2 )
{
  // size of the data
  const int N = 2;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[2*N] = {3.f, 0.f, -1.f, 0.f};

  // call the test function
  test_dft1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft1D_3 )
{
  // size of the data
  const int N = 3;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[2*N] = {6.f, 0.f, -1.5f, 0.8660f, -1.5f, -0.8660f};

  // call the test function
  test_dft1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft1D_4 )
{
  // size of the data
  const int N = 4;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[2*N] = {10.f, 0.f, -2.f, 2.f, -2.f, 0.f, -2.f, -2.f};

  // call the test function
  test_dft1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft1D_8 )
{
  // size of the data
  const int N = 8;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[2*N] = 
    {36.f, 0.f, -4.f, 9.6569f, -4.f, 4.f, -4.f, 1.6569f,
     -4.f, 0.f, -4.f,-1.6569f, -4.f, -4.f, -4.f,-9.6569f};

  // call the test function
  test_dft1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft1D_16 )
{
  // size of the data
  const int N = 16;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[2*N] = 
    {136.00f, 0.f, -8.f, 40.2187f, -8.f, 19.3137f, -8.f, 11.9728f,
     -8.f, 8.f, -8.f, 5.3454f, -8.f, 3.3137f, -8.f, 1.5913f,
     -8.f, 0.f, -8.f, -1.5913f, -8.f, -3.3137f, -8.f, -5.3454f,
     -8.f, -8.f, -8.f, -11.9728f, -8.f, -19.3137f, -8.f, -40.2187f};

  // call the test function
  test_dft1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft1D_17 )
{
  // size of the data
  const int N = 17;

  // set up simple 1D tensor
  Torch::FloatTensor t(N);
  for( int i=0; i < N; ++i)
    t.set(i, 1.0f+i);

  // array containing matlab values
  const float mat[2*N] = 
    {153.f,0.f, -8.5f,45.4710f, -8.5f,21.9410f, -8.5f,13.7280f,
     -8.5f,9.3241f, -8.5f,6.4189f, -8.5f,4.2325f, -8.5f,2.4185f,
     -8.5f,0.7876f, -8.5f,-0.7876f, -8.5f,-2.4185f, -8.5f,-4.2325f,
     -8.5f,-6.4189f, -8.5f,-9.3241f, -8.5f,-13.7280f, -8.5f,-21.9410f,
     -8.5f,-45.4710f};

  // call the test function
  test_dft1D(N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft2D_2x2a )
{
  // size of the data
  const int M = 2;
  const int N = 2;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  for( int i=0; i < M; ++i)
    for( int j=0; j < N; ++j)
      t.set(i,j, 1.0f+i+j);

  // array containing matlab values
  const float mat[M*N*2] = {8.f, 0.f, -2.f, 0.f, -2.f, 0.f, 0.f, 0.f};

  // call the test function
  test_dft2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft2D_2x2b )
{
  // size of the data
  const int M = 2;
  const int N = 2;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  t.set(0, 0, 3.2f);
  t.set(0, 1, 4.7f);
  t.set(1, 0, 5.4f);
  t.set(1, 1, 0.2f);

  // array containing matlab values
  const float mat[M*N*2] = 
    {13.5f, 0.f, 3.7f, 0.f, 2.3f, 0.f, -6.7f, 0.f};

  // call the test function
  test_dft2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft2D_3x3 )
{
  // size of the data
  const int M = 3;
  const int N = 3;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  for( int i=0; i < M; ++i)
    for( int j=0; j < N; ++j)
      t.set(i,j, 1.0f+i+j);

  // array containing matlab values
  const float mat[M*N*2] = 
    {27.f, 0.f, -4.5f, 2.5981f, -4.5f, -2.5981f,
     -4.5f, 2.5981f, 0.f, 0.f, 0.f, 0.f,
     -4.5f, -2.5981f, 0.f, 0.f, 0.f, 0.f};

  // call the test function
  test_dft2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_CASE( test_dft2D_4x4 )
{
  // size of the data
  const int M = 4;
  const int N = 4;

  // set up simple 1D tensor
  Torch::FloatTensor t(M,N);
  for( int i=0; i < M; ++i)
    for( int j=0; j < N; ++j)
      t.set(i,j, 1.0f+i+j);

  // array containing matlab values
  const float mat[M*N*2] = 
    {64.f, 0.f, -8.f, 8.f, -8.f,0.f, -8.f, -8.f, 
     -8.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     -8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
     -8.f, -8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // call the test function
  test_dft2D(M, N, t, mat, eps);
}

BOOST_AUTO_TEST_SUITE_END()
