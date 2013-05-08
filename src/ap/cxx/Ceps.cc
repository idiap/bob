/**
 * @file ap/cxx/Ceps.cc
 * @date Wed Jan 11:09:30 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <bob/ap/Ceps.h>
#include <bob/core/assert.h>
#include <bob/core/cast.h>

bob::ap::Ceps::Ceps(const double sampling_frequency,
    const double win_length_ms, const double win_shift_ms,
    const size_t n_filters, const size_t n_ceps, const double f_min,
    const double f_max, const size_t delta_win, const double pre_emphasis_coeff,
    const bool mel_scale, const bool dct_norm):
  bob::ap::Spectrogram(sampling_frequency, win_length_ms, win_shift_ms, 
    n_filters, f_min, f_max, pre_emphasis_coeff, mel_scale),
  m_n_ceps(n_ceps), m_delta_win(delta_win), m_dct_norm(dct_norm),
  m_with_energy(false), m_with_delta(false), m_with_delta_delta(false)
{
  initCacheDctKernel();
}

bob::ap::Ceps::Ceps(const bob::ap::Ceps& other):
  bob::ap::Spectrogram(other),
  m_n_ceps(other.m_n_ceps), m_delta_win(other.m_delta_win),
  m_dct_norm(other.m_dct_norm), m_with_energy(other.m_with_energy),
  m_with_delta(other.m_with_delta),
  m_with_delta_delta(other.m_with_delta_delta)
{
  initCacheDctKernel();
}

bob::ap::Ceps&
bob::ap::Ceps::operator=(const bob::ap::Ceps& other)
{
  if (this != &other)
  {
    bob::ap::Spectrogram::operator=(other);
    m_n_ceps = other.m_n_ceps;
    m_delta_win = other.m_delta_win;
    m_dct_norm = other.m_dct_norm;
    m_with_energy = other.m_with_energy;
    m_with_delta = other.m_with_delta;
    m_with_delta_delta = other.m_with_delta_delta;

    initCacheDctKernel();
  }
  return *this;
}

bool bob::ap::Ceps::operator==(const bob::ap::Ceps& other) const
{
  return (bob::ap::Spectrogram::operator==(other) && 
          m_n_ceps == other.m_n_ceps && 
          m_delta_win == other.m_delta_win &&
          m_dct_norm == other.m_dct_norm &&
          m_with_energy == other.m_with_energy &&
          m_with_delta == other.m_with_delta &&
          m_with_delta_delta == other.m_with_delta_delta);
}

bool bob::ap::Ceps::operator!=(const bob::ap::Ceps& other) const
{
  return !(this->operator==(other));
}

bob::ap::Ceps::~Ceps()
{
}

void bob::ap::Ceps::setNFilters(size_t n_filters)
{
  bob::ap::Spectrogram::setNFilters(n_filters);
  initCacheDctKernel();
}

void bob::ap::Ceps::setNCeps(size_t n_ceps)
{ 
  m_n_ceps = n_ceps; 
  initCacheFilterBank(); 
  initCacheDctKernel(); 
} 

void bob::ap::Ceps::setDctNorm(bool dct_norm)
{ 
  m_dct_norm = dct_norm;
  initCacheDctKernel();
}

void bob::ap::Ceps::initCacheDctKernel()
{
  // Dct Kernel initialization
  m_dct_kernel.resize(m_n_ceps,m_n_filters);
  blitz::firstIndex i;
  blitz::secondIndex j;
  double dct_coeff = m_dct_norm ? (double)sqrt(2./(double)(m_n_filters)) : 1.;
  m_dct_kernel = dct_coeff * blitz::cos(M_PI*(i+1)*(j+0.5)/(double)(m_n_filters));
}


blitz::TinyVector<int,2> bob::ap::Ceps::getShape(const size_t input_size) const
{
  // Res will contain the number of frames x the dimension of the feature vector
  blitz::TinyVector<int,2> res;

  // 1. Number of frames
  res(0) = 1+((input_size-m_win_length)/m_win_shift);

  // 2. Dimension of the feature vector
  int dim0=m_n_ceps;
  if (m_with_energy) dim0 += 1;
  int dim = dim0;
  if (m_with_delta)
  {
    dim += dim0;
    if(m_with_delta_delta) dim += dim0;
  }
  res(1) = dim;

  return res;
}

blitz::TinyVector<int,2> bob::ap::Ceps::getShape(const blitz::Array<double,1>& input) const
{
  return getShape(input.extent(0));
}

void bob::ap::Ceps::operator()(const blitz::Array<double,1>& input, 
  blitz::Array<double,2>& ceps_matrix)
{
  // Get expected dimensionality of output array
  blitz::TinyVector<int,2> feature_shape = bob::ap::Ceps::getShape(input);
  // Check dimensionality of output array
  bob::core::array::assertSameShape(ceps_matrix, feature_shape);
  int n_frames=feature_shape(0);

  blitz::Range r1(0,m_n_ceps-1);
  for (int i=0; i<n_frames; ++i) 
  {
    // Set padded frame to zero
    extractNormalizeFrame(input, i, m_cache_frame_d);

    // Update output with energy if required
    if (m_with_energy)
      ceps_matrix(i,(int)m_n_ceps) = logEnergy(m_cache_frame_d);

    // Apply pre-emphasis
    pre_emphasis(m_cache_frame_d);
    // Apply the Hamming window
    hammingWindow(m_cache_frame_d);
    // Filter with the triangular filter bank (either in linear or Mel domain)
    filterBank(m_cache_frame_d);
    // Apply DCT kernel and update the output 
    blitz::Array<double,1> ceps_matrix_row(ceps_matrix(i,r1));
    applyDct(ceps_matrix_row);
  }

  //compute the center of the cut-off frequencies
  const int n_coefs = (m_with_energy ?  m_n_ceps + 1 :  m_n_ceps);
  blitz::Range rall = blitz::Range::all();
  blitz::Range ro0(0,n_coefs-1);
  blitz::Range ro1(n_coefs,2*n_coefs-1);
  blitz::Range ro2(2*n_coefs,3*n_coefs-1);
  if (m_with_delta)
  {
    blitz::Array<double,2> ceps_matrix_0(ceps_matrix(rall,ro0));
    blitz::Array<double,2> ceps_matrix_1(ceps_matrix(rall,ro1));
    addDerivative(ceps_matrix_0, ceps_matrix_1);

    if (m_with_delta_delta)
    {
      blitz::Array<double,2> ceps_matrix_2(ceps_matrix(rall,ro2));
      addDerivative(ceps_matrix_1, ceps_matrix_2);
    }
  }
}

void bob::ap::Ceps::applyDct(blitz::Array<double,1>& ceps_row) const
{
  blitz::firstIndex i;
  blitz::secondIndex j;
  ceps_row = blitz::sum(m_cache_filters(j) * m_dct_kernel(i,j), j);
}

void bob::ap::Ceps::addDerivative(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const
{
  // Initialize output to zero
  output = 0.;

  const int n_frames = input.extent(0);
  blitz::Range rall = blitz::Range::all();

  // Fill in the inner part as follows:
  // \f$output[i] += \sum_{l=1}^{DW} l * (input[i+l] - input[i-l])\f$
  for (int l=1; l<=(int)m_delta_win; ++l) {
    blitz::Range rout(l,n_frames-l-1);
    blitz::Range rp(2*l,n_frames-1);
    blitz::Range rn(0,n_frames-2*l-1);
    output(rout,rall) += l*(input(rp,rall) - input(rn,rall));
  }

  const double factor = m_delta_win*(m_delta_win+1)/2;
  // Continue to fill the left boundary part as follows:
  // \f$output[i] += (\sum_{l=1+i}^{DW} l*input[i+l]) - (\sum_{l=i+1}^{DW}l)*input[0])\f$
  for (int i=0; i<(int)m_delta_win; ++i) {
    output(i,rall) -= (factor - i*(i+1)/2) * input(0,rall);
    for (int l=1+i; l<=(int)m_delta_win; ++l) {
      output(i,rall) += l*(input(i+l,rall));
    }
  }
  // Continue to fill the right boundary part as follows:
  // \f$output[i] += (\sum_{l=Nframes-1-i}^{DW}l)*input[Nframes-1]) - (\sum_{l=Nframes-1-i}^{DW} l*input[i-l])\f$
  for (int i=n_frames-(int)m_delta_win; i<n_frames; ++i) {
    int ii = (n_frames-1)-i;
    output(i,rall) += (factor - ii*(ii+1)/2) * input(n_frames-1,rall);
    for (int l=1+ii; l<=(int)m_delta_win; ++l) {
      output(i,rall) -= l*input(i-l,rall);
    }
  }
  // Sum of the integer squared from 1 to delta_win
  const double sum = m_delta_win*(m_delta_win+1)*(2*m_delta_win+1)/3;
  output /= sum;
}

/*
bob::ap::TestCeps::TestCeps(Ceps& ceps): m_ceps(ceps) {
}
*/
