/**
 * @file sp/cxx/test/fft_fct.cc
 * @date Thu Sep  5 11:32:14 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Benchmark of FFT/DFT and FCT/DCT implementations
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/core/array_random.h>
#include <bob/core/cast.h>
#include <bob/sp/FFT1DNaive.h>
#include <bob/sp/FFT1D.h>
#include <bob/sp/FFT2DNaive.h>
#include <bob/sp/FFT2D.h>
#include <bob/sp/DCT1DNaive.h>
#include <bob/sp/DCT1D.h>
#include <bob/sp/DCT2DNaive.h>
#include <bob/sp/DCT2D.h>

#include <boost/random.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>

void benchmark_fct1D(const blitz::Array<double,1> t) 
{
  const int M = t.extent(0);
  blitz::Array<double,1> t_fct(M), t_dct(M);
  boost::posix_time::ptime t1;
  boost::posix_time::ptime t2;
  boost::posix_time::time_duration diff;

  std::cout << "1D FCT/DCT on an array of dimension " << M << "..." << std::endl;

  // process using DCT
  bob::sp::DCT1D fct_numpy(M);
  t1 = boost::posix_time::microsec_clock::local_time();
  fct_numpy(t, t_fct);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  FCT duration in (microseconds) " << diff.total_microseconds() << std::endl;

  // process using DFT answer
  bob::sp::detail::DCT1DNaive dct(M);
  t1 = boost::posix_time::microsec_clock::local_time();
  dct(t, t_dct);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  DCT duration in (microseconds) " << diff.total_microseconds() << std::endl;
}

void benchmark_fct2D(const blitz::Array<double,2> t) 
{
  const int M = t.extent(0);
  const int N = t.extent(1);
  blitz::Array<double,2> t_fct(M, N), t_dct(M, N);
  boost::posix_time::ptime t1;
  boost::posix_time::ptime t2;
  boost::posix_time::time_duration diff;

  std::cout << "2D FCT/DCT on an array of dimension " << M << "x" << N << "..." << std::endl;

  // process using DCT
  bob::sp::DCT2D fct_numpy(M, N);
  t1 = boost::posix_time::microsec_clock::local_time();
  fct_numpy(t, t_fct);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  FCT duration in (microseconds) " << diff.total_microseconds() << std::endl;

  // process using DFT answer
  bob::sp::detail::DCT2DNaive dct(M, N);
  t1 = boost::posix_time::microsec_clock::local_time();
  dct(t, t_dct);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  DCT duration in (microseconds) " << diff.total_microseconds() << std::endl;
}


void benchmark_fft1D(const blitz::Array<std::complex<double>,1> t) 
{
  const int M = t.extent(0);
  blitz::Array<std::complex<double>,1> t_fft(M), t_dft(M);
  boost::posix_time::ptime t1;
  boost::posix_time::ptime t2;
  boost::posix_time::time_duration diff;

  std::cout << "1D FFT/DFT on an array of dimension " << M << "..." << std::endl;

  // process using FFT
  bob::sp::FFT1D fft_numpy(M);
  t1 = boost::posix_time::microsec_clock::local_time();
  fft_numpy(t, t_fft);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  FFT duration in (microseconds) " << diff.total_microseconds() << std::endl;

  // process using DFT answer
  bob::sp::detail::FFT1DNaive dft(M);
  t1 = boost::posix_time::microsec_clock::local_time();
  dft(t, t_dft);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  DFT duration in (microseconds) " << diff.total_microseconds() << std::endl;
}

void benchmark_fft2D(const blitz::Array<std::complex<double>,2> t) 
{
  const int M = t.extent(0);
  const int N = t.extent(1);
  blitz::Array<std::complex<double>,2> t_fft(M, N), t_dft(M, N);
  boost::posix_time::ptime t1;
  boost::posix_time::ptime t2;
  boost::posix_time::time_duration diff;

  std::cout << "2D FFT/DFT on an array of dimension " << M << "x" << N << "..." << std::endl;

  // process using FFT
  bob::sp::FFT2D fft_numpy(M, N);
  t1 = boost::posix_time::microsec_clock::local_time();
  fft_numpy(t, t_fft);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  FFT duration in (microseconds) " << diff.total_microseconds() << std::endl;

  // process using DFT answer
  bob::sp::detail::FFT2DNaive dft(M, N);
  t1 = boost::posix_time::microsec_clock::local_time();
  dft(t, t_dft);
  t2 = boost::posix_time::microsec_clock::local_time();
  diff = t2 - t1;
  std::cout << "  DFT duration in (microseconds) " << diff.total_microseconds() << std::endl;
}

/*************** FCT Tests *****************/
int main()
{
  boost::mt19937 rng(0);

  const int P=6;
  int dims[P] = {16, 64, 128, 256, 512, 1024};
  for(int i=0; i<P; ++i)
  {
    const int M = dims[i];
    // 1D array
    blitz::Array<double,1> t_d_1d(M);
    bob::core::array::randn(rng, t_d_1d);
    // Benchmark
    benchmark_fct1D(t_d_1d);
  }

  for(int i=0; i<P; ++i)
  {
    const int M = dims[i];
    // 2D array
    blitz::Array<double,2> t_d_2d(M,M);
    bob::core::array::randn(rng, t_d_2d);
    // Benchmark
    benchmark_fct2D(t_d_2d);
  }

  for(int i=0; i<P; ++i)
  {
    const int M = dims[i];
    // 1D array
    blitz::Array<double,1> t_d_1d(M);
    bob::core::array::randn(rng, t_d_1d);
    blitz::Array<std::complex<double>,1> t_1d = bob::core::array::cast<std::complex<double> >(t_d_1d);
    // Benchmark
    benchmark_fft1D(t_1d);
  }

  for(int i=0; i<P; ++i)
  {
    const int M = dims[i];
    // 2D array
    blitz::Array<double,2> t_d_2d(M,M);
    bob::core::array::randn(rng, t_d_2d);
    blitz::Array<std::complex<double>,2> t_2d = bob::core::array::cast<std::complex<double> >(t_d_2d);
    // Benchmark
    benchmark_fft2D(t_2d);
  }

  return 0;
}
