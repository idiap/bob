/**
 * @file ap/cxx/Ceps.cc
 * @date Wed Jan 11:09:30 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 *
 * @brief Implement Linear and Mel Frequency Cepstral Coefficients
 * functions (MFCC and LFCC)
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/ap/Ceps.h"
#include "bob/core/array_assert.h"
#include "bob/core/cast.h"
#include "bob/sp/FFT1D.h"
#include <cmath>
#include <blitz/array.h>

bob::ap::TestCeps::TestCeps(Ceps& ceps): m_ceps(ceps) {
}

bob::ap::Ceps::Ceps( double sf, int win_length_ms, int win_shift_ms, int n_filters, int n_ceps,
    double f_min, double f_max, double delta_win):
  m_sf(sf), m_win_length_ms(win_length_ms), m_win_shift_ms(win_shift_ms), 
  m_n_filters(n_filters), m_n_ceps(n_ceps), 
  m_f_min(f_min), m_f_max(f_max), m_delta_win(delta_win),
  m_fb_linear(true), m_dct_norm(1.), m_with_energy(true), m_with_delta(true), m_with_delta_delta(true), 
  m_with_delta_energy(true), m_with_delta_delta_energy(true)
{
  initWinLength();
  initWinShift();
  initWinSize();
  initCache();
}

bob::ap::Ceps::~Ceps()
{
}

void bob::ap::Ceps::setSampleFrequency(const double sf) 
{ 
  m_sf = sf; 
  initWinLength();
  initWinShift();
}

void bob::ap::Ceps::setWinLengthMs(int win_length_ms)
{ 
  m_win_length_ms = win_length_ms;
  initWinLength(); 
}

void bob::ap::Ceps::setWinShiftMs(int win_shift_ms)
{ 
  m_win_shift_ms = win_shift_ms;
  initWinShift(); 
}

void bob::ap::Ceps::setNFilters(int n_filters)
{ 
  m_n_filters = n_filters; 
  m_filters.resize(m_n_filters); 
  initCachePIndex(); 
  initCacheDctKernel(); 
}

void bob::ap::Ceps::setNCeps(int n_ceps)
{ 
  m_n_ceps = n_ceps; 
  m_ceps_coeff.resize(m_n_ceps);
  initCachePIndex(); 
  initCacheDctKernel(); 
} 

void bob::ap::Ceps::setFMin(double f_min)
{ 
  m_f_min = f_min; 
  initCachePIndex(); 
}

void bob::ap::Ceps::setFMax(double f_max)
{ 
  m_f_max = f_max; 
  initCachePIndex();
}

void bob::ap::Ceps::setFbLinear(bool fb_linear)
{ 
  m_fb_linear = fb_linear; 
  initCachePIndex(); 
}

//Auxilary functions needed to set mel scale
double bob::ap::Ceps::mel(double f)
{
  return(2595.*log10(1+f/700.));
}
double bob::ap::Ceps::MelInv(double f)
{
  return((double)(700.*(pow(10,f/2595.)-1)));
}

void bob::ap::Ceps::initWinLength()
{ 
  m_win_length = (int)(m_sf * m_win_length_ms / 1000); 
  initWinSize();
  initCacheHammingKernel(); 
  initCachePIndex(); 
}

void bob::ap::Ceps::initWinShift()
{ 
  m_win_shift = (int)(m_sf * m_win_shift_ms / 1000);
}

void bob::ap::Ceps::initWinSize()
{
  m_win_size = (int)pow(2.0,(double)ceil(log(m_win_length)/log(2)));
  m_frame.resize(m_win_size);
}

void bob::ap::Ceps::reinit(double dct_norm, bool fb_linear, bool with_energy,
    bool with_delta, bool with_delta_delta, bool with_delta_energy, bool with_delta_delta_energy)
{
  m_dct_norm = dct_norm;
  m_fb_linear = fb_linear;
  m_with_energy = with_energy;
  m_with_delta = with_delta ;
  m_with_delta_delta = with_delta_delta;
  m_with_delta_energy = with_delta_energy;
  m_with_delta_delta_energy = with_delta_delta_energy;

  initWinSize();
  initCache();
}

void bob::ap::Ceps::initCacheHammingKernel()
{
  // Hamming initialization
  m_hamming_kernel.resize(m_win_length);
  double cst = 2*M_PI/(m_win_length-1);
  blitz::firstIndex i;
  m_hamming_kernel = 0.54-0.46*blitz::cos(i*cst);
}

void bob::ap::Ceps::initCacheDctKernel()
{
  // Dct Kernel initialization
  m_dct_kernel.resize(m_n_ceps,m_n_filters);
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_dct_kernel = blitz::cos(M_PI*(i+1)*(j+0.5)/(double)(m_n_filters));
  //m_dct_norm=(double)sqrt(2.0/(double)(m_n_filters));
}

void bob::ap::Ceps::initCachePIndex()
{
  m_p_index.resize(m_n_filters+2);
  blitz::Array<double,1> p_index_d(m_n_filters+2);
  if(m_fb_linear) {
    // Linear scale
    //blitz::firstIndex i;
    //m_p_index = blitz::round(m_win_size/m_sf * (m_f_min + (m_f_max-m_f_min)*i/(double)(m_n_filters+1)));
    for(int i=0; i<(int)m_n_filters+2; ++i) {
      double alpha = (double)(i)/ (double)(m_n_filters+1);
      double f = m_f_min + (m_f_max - m_f_min) * alpha;
      m_p_index(i) = (int)round(m_win_size/m_sf * f);
    }
  } else {
    // Mel scale
    double m_max = mel(m_f_max);
    double m_min = mel(m_f_min);
    for(int i=0; i<(int)m_n_filters+2; ++i) {
      double alpha = (double) (i)/ (double) (m_n_filters+1);
      double f = MelInv(m_min * (1-alpha) + m_max * alpha);
      double factor = f / m_sf;
      m_p_index(i)=(int)round(m_win_size * factor);
    }
  }
}

void bob::ap::Ceps::initCache()
{
  m_filters.resize(m_n_filters);
  m_ceps_coeff.resize(m_n_ceps);

  initCacheHammingKernel();
  initCacheDctKernel();
  initCachePIndex();
}

blitz::TinyVector<int,2> bob::ap::Ceps::getCepsShape(const size_t input_size) const
{
  // Res will contain the number of frames x the dimension of the feature vector
  blitz::TinyVector<int,2> res;

  // 1. Number of frames
  res(0) = 1+((input_size-(int)(m_win_length))/(int)(m_win_shift));

  // 2. Dimension of the feature vector
  int dim=m_n_ceps;
  if(m_with_energy)
    dim = m_n_ceps + 1;
  if(m_with_delta)
    dim = dim + m_n_ceps;
  if(m_with_delta_energy)
    dim = dim + 1;
  if(m_with_delta_delta)
    dim = dim + m_n_ceps;
  if(m_with_delta_delta_energy)
    dim = dim + 1;
  res(1) = dim;

  return res;
}

blitz::TinyVector<int,2> bob::ap::Ceps::getCepsShape(const blitz::Array<double,1>& input) const
{
  return this->getCepsShape(input.extent(0));
}

void bob::ap::Ceps::CepsAnalysis(const blitz::Array<double,1>& input, 
  blitz::Array<double,2>& ceps_matrix)
{
  const int d1 = (m_with_energy ? m_n_ceps + 1 : m_n_ceps);
  // Get expected dimensionality of output array
  blitz::TinyVector<int,2> feature_shape = bob::ap::Ceps::getCepsShape(input);
  // Check dimensionality of output array
  bob::core::array::assertSameShape(ceps_matrix, feature_shape);
  int n_frames=feature_shape(0);

  //compute the center of the cut-off frequencies
  blitz::Range r1(0,m_n_ceps-1);
  for(int i=0; i<n_frames; ++i) {
    // Create a frame
    double sum=0.;
    m_frame = 0.;
    for(int j=0; j<m_win_length; ++j)
      m_frame(j) = input(j+i*m_win_shift);
    sum = blitz::sum(m_frame) / (double)m_win_size;
    m_frame -= sum;

    // Update output
    if(m_with_energy)
      ceps_matrix(i,(int)m_n_ceps) = logEnergy(m_frame);

    emphasis(m_frame, 0.95);
    hammingWindow(m_frame);

    // Apply the transformation
    logFilterBank(m_frame);
    transformDCT();

    // Update output
    blitz::Array<double,1> ceps_matrix_row(ceps_matrix(i,r1));
    ceps_matrix_row = m_ceps_coeff;
  }
  if (m_with_delta)
    addDelta(ceps_matrix, 2, n_frames, d1);
  if (m_with_delta_delta)
    addDeltaDelta(ceps_matrix, 2, n_frames, 2*d1);
}

void bob::ap::Ceps::emphasis(blitz::Array<double,1> &data, double a)
{
  if(a < 0. || a >= 1.0) {
    // TODO
    printf("Invalid emphasis coefficient %.2f (should be between 0 and 1)\n",a);
  }
  if(a!=0.)
  { 
    double v0 = (1.-a)*data(0);
    blitz::Range r0(m_win_length-2,0,-1); 
    blitz::Range r1(m_win_length-1,1,-1); 
    data(r1) -= a*data(r0);
    data(0) = v0;
  }
}

void bob::ap::Ceps::hammingWindow(blitz::Array<double,1> &data)
{
  blitz::Range r(0,m_win_length-1);
  data(r) *= m_hamming_kernel;
}

void bob::ap::Ceps::logFilterBank(blitz::Array<double,1>& x)
{
  int win_size = x.shape()[0];
  blitz::Array<std::complex<double>,1> x1(win_size);

  x1 = bob::core::cast<std::complex<double> >(x);

  blitz::Array<std::complex<double>,1> complex_(win_size);
  bob::sp::FFT1D fft(win_size);
  fft(x1,complex_);

  blitz::Range r(0,win_size/2);
  blitz::Array<double,1> x_half(x(r));
  blitz::Array<std::complex<double>,1> complex_half(complex_(r));
  x_half = blitz::abs(complex_half);

  logTriangularFBank(x);
}


/* -------------------------------------------------------------------- */
/* ----- void LogTriangularFBank(vector_t *,int,int *,vector_t *) ----- */
/* -------------------------------------------------------------------- */
/*
 * Apply triangular filter bank to module vector and return the log of
 * the energy in each band. Table m_p_index contains the indexes of the
 * cut-off frequencies. Looks like something like this:
 *
 *                      filter 2
 *                   <------------->
 *                filter 1           filter 3
 *             <----------->       <------------->
 *        | | | | | | | | | | | | | | | | | | | | | ..........
 *         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9  ..........
 *             ^     ^     ^       ^             ^
 *             |     |     |       |             |
 *          p_in[0]  |  p_in[2]    |          p_in[4]
 *                p_in[1]       p_in[3]
 *
 */
void bob::ap::Ceps::logTriangularFBank(blitz::Array<double,1>& data)
{
  int j;
  double a;
  double res;

  for(int i=0; i<(int)m_n_filters; ++i)
  {
    res=0.0;

    a=1.0/(double)(m_p_index(i+1)-m_p_index(i)+1);

    for(j=m_p_index(i); j<m_p_index(i+1); ++j)
    {
      res += data(j)*(1.0-a*(m_p_index(i+1)-j));
    }
    a=1.0/(double)(m_p_index(i+2)-m_p_index(i+1)+1);

    for(j=m_p_index(i+1); j<=m_p_index(i+2); ++j)
    {
      res += data(j)*(1.0-a*((double)(j-m_p_index(i+1))));
    }
    m_filters(i)=(res < FBANK_OUT_FLOOR)?(double)log(FBANK_OUT_FLOOR):(double)log(res);
  }
}

double bob::ap::Ceps::logEnergy(blitz::Array<double,1> &data)
{
  blitz::Array<double,1> data_p(data(blitz::Range(0,m_win_length-1)));
  double gain = blitz::sum(blitz::pow2(data_p));
  gain = gain < ENERGY_FLOOR ?
      (double)(log(ENERGY_FLOOR)) : (double)(log(gain));
  return (gain);
}


/* -------------------------------------------------------- */
/*
 * Apply a p order DCT to vector v1.
 * Results are returned through v2.
 *
 * If {m[1],...,m[N]} are the output of the filters, then
 *    c[i]=sqrt(2/N)*sum for j=1 to N of(m[j]cos(M_PI*i*(j-0.5)/N) i=1,...,p
 *
 * This is what is implemented here with arrays indexed from 0 to N-1.
 *
 */
void bob::ap::Ceps::transformDCT()
{
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_ceps_coeff = blitz::sum(m_filters(j) * m_dct_kernel(i,j), j);
}

void bob::ap::Ceps::addDelta(blitz::Array<double,2>& frames, int m_delta_win, int n_frames, int frame_size)
{
  // Sum of the integer squared from 1 to m_delta_win
  int sum = m_delta_win*(m_delta_win+1)*(2*m_delta_win+1)/3;

  for(int i=0; i<n_frames; ++i) {
    int k = frame_size;
    for(int j=0; j<frame_size; ++j, ++k) {
      frames(i,k) = 0.;
      for(int l=1; l<=m_delta_win; ++l) {
        int p_index = (i+l) < n_frames ? i+l : n_frames-1;
        int n_index = (i-l) > 0 ? (i-l) : 0;
        frames(i,k) += l*(frames(p_index,j)-frames(n_index,j));
      }
      frames(i,k) /=sum;
    }
  }
}

void bob::ap::Ceps::addDeltaDelta(blitz::Array<double,2>& frames, int m_delta_win, int n_frames, int frame_size) 
{
  // Sum of the integer squared from 1 to m_delta_win
  int sum = m_delta_win*(m_delta_win+1)*(2*m_delta_win+1)/3;

  for(int i=0; i<n_frames; ++i) {
    int k = frame_size;
    for(int j=frame_size/2; j<frame_size; ++j, ++k){
      frames(i,k) = 0.;
      for(int l=1; l<=m_delta_win; ++l){
        int p_index = (i+l) < n_frames ? i+l : n_frames-1;
        int n_index = (i-l) > 0 ? (i-l) : 0;
        frames(i,k) += l*(frames(p_index,j) - frames(n_index,j));
      }
      frames(i,k) /=sum;
    }
  }
}

blitz::Array<double,2> bob::ap::Ceps::dataZeroMean(blitz::Array<double,2>& frames, bool norm_energy, int n_frames, int frame_size) 
{
  blitz::Array<double,1> mean(frame_size);
  for(int i=0; i<n_frames; ++i) {
    for(int j=0; j<frame_size; ++j)
      mean(j) += frames(i,j);
  }

  for(int j=0; j<frame_size; ++j)
    mean(j) /= n_frames;

  if(!norm_energy)
    mean(frame_size) = 0.0;

  for(int i=0; i<n_frames; ++i){

    for(int j=0;j<frame_size; ++j)
      frames(i,j) -= mean(j);
  }
  return frames;
}
