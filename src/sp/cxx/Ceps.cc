/**
 * @file sp/cxx/Ceps.cc
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

#include "bob/sp/Ceps.h"
#include "bob/core/array_assert.h"
#include "bob/core/cast.h"
#include "bob/sp/FFT1D.h"
#include <cmath>
#include <blitz/array.h>

bob::sp::TestCeps::TestCeps(Ceps& ceps): m_ceps(ceps) {
}

bob::sp::Ceps::Ceps( double sf, int win_length_ms, int win_shift_ms, int n_filters, int n_ceps,
    double f_min, double f_max, double delta_win, const blitz::Array<double,1>& data_array):
                m_sf(sf), m_win_length((int) (sf * win_length_ms / 1000)), m_win_shift((int) (sf * win_shift_ms / 1000)),
                m_nfilters(n_filters), m_nceps(n_ceps), m_f_min(f_min), m_f_max(f_max), m_delta_win(delta_win), m_data_array(data_array),
                m_fb_linear(true), m_dct_norm(1.), m_withEnergy(true), m_withDelta(true), m_withDeltaDelta(true), m_withDeltaEnergy(true), m_withDeltaDeltaEnergy(true)
{
  initWindowSize();
  initCache();
}

bob::sp::Ceps::~Ceps()
{
}


//Auxilary functions needed to set mel scale
double bob::sp::Ceps::mel(double f)
{
  return(2595.0*log10(1+f/700.0));
}
double bob::sp::Ceps::MelInv(double f)
{
  return((double)(700.0*(pow(10,f/2595.0)-1)));
}

void bob::sp::Ceps::initWindowSize()
{
  m_win_size = (int)pow(2.0,(double)ceil(log(m_win_length)/log(2)));
}

void bob::sp::Ceps::reinit(double dct_norm, bool fb_linear, bool withEnergy,
    bool withDelta, bool withDeltaDelta, bool withDeltaEnergy, bool withDeltaDeltaEnergy)
{
  m_dct_norm = dct_norm;
  m_fb_linear = fb_linear;
  m_withEnergy = withEnergy;
  m_withDelta = withDelta ;
  m_withDeltaDelta = withDeltaDelta;
  m_withDeltaEnergy = withDeltaEnergy;
  m_withDeltaDeltaEnergy = withDeltaDeltaEnergy;

  initWindowSize();
  initCache();
}

void bob::sp::Ceps::initCache()
{
  m_dct_kernel.resize(m_nceps,m_nfilters);
  m_hamming_kernel.resize(m_win_length);
  m_frame.resize(m_win_size);
  m_filters.resize(m_nfilters);
  m_ceps_coeff.resize(m_nceps);
  m_p_index.resize(m_nfilters+2);

  //hamming initialization
  double cst = 2*M_PI/(m_win_length-1);

  for(int i=0;i<m_win_length; ++i)
    m_hamming_kernel(i) = (double)(0.54-0.46*cos(i*cst));

  if(m_fb_linear) {
    //Linear scale
    for(int i=0; i<m_nfilters+2; ++i) {
      double alpha = (double) (i)/ (double) (m_nfilters+1);
      double f = m_f_min + (m_f_max - m_f_min) * alpha;
      m_p_index(i) = (int)rint(m_win_size/m_sf * f);
    }
  } else {
    //Mel scale
    double m_max = mel(m_f_max);
    double m_min = mel(m_f_min);
    for(int i=0; i<m_nfilters+2; ++i) {
      double alpha = (double) (i)/ (double) (m_nfilters+1);
      double f = MelInv(m_min * (1-alpha) + m_max * alpha);
      double factor = f / m_sf;
      m_p_index(i)=(int)rint(m_win_size * factor);
    }
  }

  //cosine transform initialization
  //m_dct_norm=(double)sqrt(2.0/(double)(m_nfilters));
  for(int i=1; i<=m_nceps; ++i) {
    for(int j=1; j<=m_nfilters; ++j)
      m_dct_kernel(i-1,j-1)=(double)cos(M_PI*i*(j-0.5)/(double)(m_nfilters));
  }
}


blitz::TinyVector<int,2> bob::sp::Ceps::getCepsShape(int n_size) const
{
  // Res will contain the number of frames x the dimension of the feature vector
  blitz::TinyVector<int,2> res;

  // 1. Number of frames
  res(0) = 1+((n_size-(int)(m_win_length))/(int)(m_win_shift));

  // 2. Dimension of the feature vector
  int dim=m_nceps;
  if (m_withEnergy)
    dim = m_nceps + 1;
  if (m_withDelta)
    dim = dim + m_nceps;
  if (m_withDeltaEnergy)
    dim = dim + 1;
  if (m_withDeltaDelta)
    dim = dim + m_nceps;
  if(m_withDeltaDeltaEnergy)
    dim = dim + 1;
  res(1) = dim;

  return res;
}

void bob::sp::Ceps::CepsAnalysis(int n_size, blitz::Array<double,2>& ceps_matrix)
{
  int d1=(m_withEnergy ? m_nceps + 1 : m_nceps);
  // Get expected dimensionality of output array
  blitz::TinyVector<int,2> feature_shape = bob::sp::Ceps::getCepsShape(n_size);
  // Check dimensionality of output array
  bob::core::array::assertSameShape(ceps_matrix, feature_shape);
  int n_frames=feature_shape(0);

  //compute the center of the cut-off frequencies
  double som=0.0;
  int j = 0;
  for(int i=0;i<(int)n_frames; ++i) {
    //create a frame
    som=0.0;
    j = 0;
    for(;j<m_win_length; ++j)
    {
      m_frame(j) = m_data_array(j+i*m_win_shift);
      som += m_frame(j);
    }

    for(;j<m_win_size; ++j)
    {
      m_frame(j) = 0.0;
    }
    som = som/ (double)m_win_size;
    for(j=0;j<m_win_size; ++j)
    {
      m_frame(j) = m_frame(j) - som;
    }

    double energy=0.;
    if (m_withEnergy)
      energy = logEnergy(m_frame);

    emphasis(m_frame, m_win_length, 0.95);
    hammingWindow(m_frame);

    // Apply the transformation
    logFilterBank(m_frame);

    transformDCT();

    for(int k=0; k<m_nceps; ++k)
      ceps_matrix(i,k)=m_ceps_coeff(k);

    if (m_withEnergy)
      ceps_matrix(i,m_nceps)=energy;
  }
  if (m_withDelta)
    addDelta(ceps_matrix, 2, n_frames, d1);
  if (m_withDeltaDelta)
    addDeltaDelta(ceps_matrix, 2, n_frames, 2*d1);
}

void bob::sp::Ceps::emphasis(blitz::Array<double,1> &data, int n,double a)
{
  if(a < 0. || a >= 1.0) {
    // TODO
    printf("Invalid emphasis coefficient %.2f (should be between 0 and 1)\n",a);
  }
  if(a!=0.)
  {
    for(int i=n-1;i>0;i--)
      data(i) = data(i)-a*data(i-1);
    data(0) = (1. - a)*data(0);
  }
}

void bob::sp::Ceps::hammingWindow(blitz::Array<double,1> &data)
{
  for(int i=0;i<m_win_length;i++)
    data(i) *= m_hamming_kernel(i);
}

void bob::sp::Ceps::logFilterBank(blitz::Array<double,1>& x)
{
  int win_size = x.shape()[0];
  blitz::Array<std::complex<double>,1> x1(win_size);

  x1 = bob::core::cast<std::complex<double> >(x);

  blitz::Array<std::complex<double>,1> complex_(win_size);
  bob::sp::FFT1D fft(win_size);
  fft(x1,complex_);

  int sh = win_size/2 ;
  blitz::Array<double,1> x_half(x(blitz::Range(0,sh)));
  blitz::Array<std::complex<double>,1> complex_half(complex_(blitz::Range(0,sh)));
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
void bob::sp::Ceps::logTriangularFBank(blitz::Array<double,1>& data)
{
  int j;
  double a;
  double res;

  for(int i=0; i<m_nfilters; ++i)
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

double bob::sp::Ceps::logEnergy(blitz::Array<double,1> &data)
{
  double gain=0.;
  for(int i=0; i<m_win_length; ++i)
    gain+=data(i)*data(i);

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
void bob::sp::Ceps::transformDCT()
{
  for(int i=1; i<=m_nceps; ++i) {
    m_ceps_coeff(i-1)=0.0;
    for(int j=1;j<=m_nfilters; ++j)
      m_ceps_coeff(i-1) += (m_filters(j-1)*m_dct_kernel(i-1,j-1));
    m_ceps_coeff(i-1) *= (double)m_dct_norm;
  }
}

void bob::sp::Ceps::addDelta(blitz::Array<double,2>& frames, int m_delta_win, int n_frames, int frame_size)
{
  int sum = 0;
  for(int i=1; i<=m_delta_win; ++i)
    sum += i*i;
  sum *=2;

  for(int i=0; i<n_frames; ++i){
    int k = frame_size;
    for(int j=0; j<frame_size; ++j, ++k) {
      frames(i,k) = 0.0;
      for(int l=1; l<=m_delta_win; ++l){
        int p_index = (i+l) < n_frames ? i+l : n_frames-1;
        int n_index = (i-l) > 0 ? (i-l) : 0;
        frames(i,k) += l*(frames(p_index,j)-frames(n_index,j));
      }
      frames(i,k) /=sum;
    }
  }
}

void bob::sp::Ceps::addDeltaDelta(blitz::Array<double,2>& frames, int m_delta_win, int n_frames, int frame_size) {
  int sum = 0;
  for(int i=1; i<=m_delta_win; ++i)
    sum += i*i;
  sum *=2;

  for(int i=0; i<n_frames; ++i){

    int k = frame_size;
    for(int j=frame_size/2; j<frame_size; ++j, ++k){
      frames(i,k) = 0.0;
      for(int l=1; l<=m_delta_win; ++l){
        int p_index = (i+l) < n_frames ? i+l : n_frames-1;
        int n_index = (i-l) > 0 ? (i-l) : 0;
        frames(i,k) += l*(frames(p_index,j) - frames(n_index,j));
      }
      frames(i,k) /=sum;
    }
  }
}

blitz::Array<double,2> bob::sp::Ceps::dataZeroMean(blitz::Array<double,2>& frames, bool norm_energy, int n_frames, int frame_size) {
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
