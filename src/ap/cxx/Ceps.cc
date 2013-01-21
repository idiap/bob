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

bob::ap::Ceps::Ceps( double sampling_frequency, double win_length_ms, double win_shift_ms,
    size_t n_filters, size_t n_ceps, double f_min, double f_max, 
    size_t delta_win, double pre_emphasis_coeff, bool mel_scale,
    bool dct_norm):
  m_sampling_frequency(sampling_frequency), m_win_length_ms(win_length_ms), m_win_shift_ms(win_shift_ms),
  m_n_filters(n_filters), m_n_ceps(n_ceps), m_f_min(f_min), m_f_max(f_max), 
  m_delta_win(delta_win), m_pre_emphasis_coeff(pre_emphasis_coeff),
  m_mel_scale(mel_scale), m_dct_norm(dct_norm),
  m_with_energy(false), m_with_delta(false), m_with_delta_delta(false),
  m_energy_floor(1.), m_fb_out_floor(1.), m_fft(1)
{
  initWinLength();
  initWinShift();

  // Initializes logarithm of flooring values
  m_log_energy_floor = log(m_energy_floor);
  m_log_fb_out_floor = log(m_fb_out_floor);

  m_cache_filters.resize(m_n_filters);
  initCacheDctKernel();
}

bob::ap::Ceps::~Ceps()
{
}

void bob::ap::Ceps::setSamplingFrequency(const double sampling_frequency)
{ 
  m_sampling_frequency = sampling_frequency;
  initWinLength();
  initWinShift();
}

void bob::ap::Ceps::setWinLengthMs(double win_length_ms)
{ 
  m_win_length_ms = win_length_ms;
  initWinLength(); 
}

void bob::ap::Ceps::setWinShiftMs(double win_shift_ms)
{ 
  m_win_shift_ms = win_shift_ms;
  initWinShift(); 
}

void bob::ap::Ceps::setNFilters(size_t n_filters)
{ 
  m_n_filters = n_filters; 
  m_cache_filters.resize(m_n_filters); 
  initCacheFilterBank(); 
  initCacheDctKernel(); 
}

void bob::ap::Ceps::setNCeps(size_t n_ceps)
{ 
  m_n_ceps = n_ceps; 
  initCacheFilterBank(); 
  initCacheDctKernel(); 
} 

void bob::ap::Ceps::setFMin(double f_min)
{ 
  m_f_min = f_min; 
  initCacheFilterBank(); 
}

void bob::ap::Ceps::setFMax(double f_max)
{ 
  m_f_max = f_max; 
  initCacheFilterBank();
}

void bob::ap::Ceps::setMelScale(bool mel_scale)
{ 
  m_mel_scale = mel_scale;
  initCacheFilterBank(); 
}

void bob::ap::Ceps::setDctNorm(bool dct_norm)
{ 
  m_dct_norm = dct_norm;
  initCacheDctKernel();
}

double bob::ap::Ceps::herzToMel(double f)
{
  return(2595.*log10(1+f/700.));
}

double bob::ap::Ceps::melToHerz(double f)
{
  return((double)(700.*(pow(10,f/2595.)-1)));
}

void bob::ap::Ceps::initWinLength()
{ 
  m_win_length = (size_t)(m_sampling_frequency * m_win_length_ms / 1000);
  initWinSize();
  initCacheHammingKernel(); 
  initCacheFilterBank(); 
}

void bob::ap::Ceps::initWinShift()
{ 
  m_win_shift = (size_t)(m_sampling_frequency * m_win_shift_ms / 1000);
}

void bob::ap::Ceps::initWinSize()
{
  m_win_size = (size_t)pow(2.0,ceil(log((double)m_win_length)/log(2)));
  m_cache_frame_d.resize(m_win_size);
  m_fft.reset(m_win_size);
  m_cache_frame_c1.resize(m_win_size);
  m_cache_frame_c2.resize(m_win_size);
}

void bob::ap::Ceps::initCacheHammingKernel()
{
  // Hamming Window initialization
  m_hamming_kernel.resize(m_win_length);
  double cst = 2*M_PI/(double)(m_win_length-1);
  blitz::firstIndex i;
  m_hamming_kernel = 0.54-0.46*blitz::cos(i*cst);
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

void bob::ap::Ceps::initCacheFilterBank()
{
  initCachePIndex();
  initCacheFilters();
}

void bob::ap::Ceps::initCachePIndex()
{
  // Computes the indices for the triangular filter bank
  m_p_index.resize(m_n_filters+2);
  // 'Mel' frequency decomposition (for MFCC)
  if(m_mel_scale)
  {
    double m_max = herzToMel(m_f_max);
    double m_min = herzToMel(m_f_min);
    for(int i=0; i<(int)m_n_filters+2; ++i) {
      double alpha = i/ (double)(m_n_filters+1);
      double f = melToHerz(m_min * (1-alpha) + m_max * alpha);
      double factor = f / m_sampling_frequency;
      m_p_index(i)=(int)round(m_win_size * factor);
    }
  }

  else
  // Linear frequency decomposition (for LFCC)
  {
    const double cst_a = (m_win_size/m_sampling_frequency) * (m_f_max-m_f_min)/(double)(m_n_filters+1);
    const double cst_b = (m_win_size/m_sampling_frequency) * m_f_min;
    for(int i=0; i<(int)m_n_filters+2; ++i) {
      m_p_index(i) = (int)round(cst_a * i + cst_b);
    }
  }
}

void bob::ap::Ceps::initCacheFilters()
{
  // Creates the Triangular filter bank
  m_filter_bank.clear();
  blitz::firstIndex ii;
  for(int i=0; i<(int)m_n_filters; ++i) 
  {
    // Integer indices of the boundary of the triangular filter in the 
    // Fourier domain
    int li = m_p_index(i);
    int mi = m_p_index(i+1);
    int ri = m_p_index(i+2);
    blitz::Array<double,1> filt(ri-li+1);
    // Fill in the left slice of the triangular filter
    blitz::Array<double,1> filt_p1(filt(blitz::Range(0,mi-li-1)));
    int len = mi-li+1;
    double a = 1. / len;
    filt_p1 = 1.-a*(len-1-ii);
    // Fill in the right slice of the triangular filter
    blitz::Array<double,1> filt_p2(filt(blitz::Range(mi-li,ri-li)));
    len = ri-mi+1;
    a = 1. / len;
    filt_p2 = 1.-a*ii;
    // Append filter into the filterbank vector
    m_filter_bank.push_back(filt);
  }
}

blitz::TinyVector<int,2> bob::ap::Ceps::getCepsShape(const size_t input_size) const
{
  // Res will contain the number of frames x the dimension of the feature vector
  blitz::TinyVector<int,2> res;

  // 1. Number of frames
  res(0) = 1+((input_size-m_win_length)/m_win_shift);

  // 2. Dimension of the feature vector
  int dim0=m_n_ceps;
  if(m_with_energy) dim0 += 1;
  int dim = dim0;
  if(m_with_delta)
  {
    dim += dim0;
    if(m_with_delta_delta) dim += dim0;
  }
  res(1) = dim;

  return res;
}

blitz::TinyVector<int,2> bob::ap::Ceps::getCepsShape(const blitz::Array<double,1>& input) const
{
  return getCepsShape(input.extent(0));
}

void bob::ap::Ceps::operator()(const blitz::Array<double,1>& input, 
  blitz::Array<double,2>& ceps_matrix)
{
  // Get expected dimensionality of output array
  blitz::TinyVector<int,2> feature_shape = bob::ap::Ceps::getCepsShape(input);
  // Check dimensionality of output array
  bob::core::array::assertSameShape(ceps_matrix, feature_shape);
  int n_frames=feature_shape(0);

  //compute the center of the cut-off frequencies
  const int n_coefs = (m_with_energy ?  m_n_ceps + 1 :  m_n_ceps);
  blitz::Range r1(0,m_n_ceps-1);
  blitz::Range rf(0,(int)m_win_length-1); 
  for(int i=0; i<n_frames; ++i) 
  {
    // Set padded frame to zero
    m_cache_frame_d = 0.;
    // Extract frame input vector
    blitz::Range ri(i*(int)m_win_shift,i*(int)m_win_shift+(int)m_win_length-1);
    m_cache_frame_d(rf) = input(ri);
    // Substract mean value
    m_cache_frame_d -= blitz::mean(m_cache_frame_d);

    // Update output with energy if required
    if(m_with_energy)
      ceps_matrix(i,(int)m_n_ceps) = logEnergy(m_cache_frame_d);

    // Apply pre-emphasis
    pre_emphasis(m_cache_frame_d);
    // Apply the Hamming window
    hammingWindow(m_cache_frame_d);
    // Filter with the triangular filter bank (either in linear or Mel domain)
    logFilterBank(m_cache_frame_d);
    // Apply DCT kernel and update the output 
    blitz::Array<double,1> ceps_matrix_row(ceps_matrix(i,r1));
    applyDct(ceps_matrix_row);
  }

  blitz::Range rall = blitz::Range::all();
  blitz::Range ro0(0,n_coefs-1);
  blitz::Range ro1(n_coefs,2*n_coefs-1);
  blitz::Range ro2(2*n_coefs,3*n_coefs-1);
  if(m_with_delta)
  {
    blitz::Array<double,2> ceps_matrix_0(ceps_matrix(rall,ro0));
    blitz::Array<double,2> ceps_matrix_1(ceps_matrix(rall,ro1));
    addDerivative(ceps_matrix_0, ceps_matrix_1);

    if(m_with_delta_delta)
    {
      blitz::Array<double,2> ceps_matrix_2(ceps_matrix(rall,ro2));
      addDerivative(ceps_matrix_1, ceps_matrix_2);
    }
  }
}

void bob::ap::Ceps::pre_emphasis(blitz::Array<double,1> &data) const
{
  if(m_pre_emphasis_coeff!=0.)
  { 
    // Pre-emphasise the signal by applying the first order equation
    // \f$data_{n} := data_{n} − a*data_{n−1}\f$
    blitz::Range r0((int)m_win_length-2,0,-1); 
    blitz::Range r1((int)m_win_length-1,1,-1); 
    data(r1) -= m_pre_emphasis_coeff * data(r0); // Apply first order equation
    data(0) *= 1. - m_pre_emphasis_coeff; // Update first element
  }
}

void bob::ap::Ceps::hammingWindow(blitz::Array<double,1> &data) const
{
  blitz::Range r(0,(int)m_win_length-1);
  data(r) *= m_hamming_kernel;
}

void bob::ap::Ceps::logFilterBank(blitz::Array<double,1>& x)
{
  // Apply the FFT
  m_cache_frame_c1 = bob::core::cast<std::complex<double> >(x);
  m_fft(m_cache_frame_c1, m_cache_frame_c2);

  // Take the the power spectrum of the first part of the output of the FFT
  blitz::Range r(0,(int)m_win_size/2);
  blitz::Array<double,1> x_half(x(r));
  blitz::Array<std::complex<double>,1> complex_half(m_cache_frame_c2(r));
  x_half = blitz::abs(complex_half);

  // Apply the Triangular filter bank to this power spectrum
  logTriangularFilterBank(x);
}

void bob::ap::Ceps::logTriangularFilterBank(blitz::Array<double,1>& data) const
{
  for(int i=0; i<(int)m_n_filters; ++i)
  {
    blitz::Array<double,1> data_slice(data(blitz::Range(m_p_index(i),m_p_index(i+2))));
    double res = blitz::sum(data_slice * m_filter_bank[i]);
    m_cache_filters(i)= (res < m_fb_out_floor ? m_log_fb_out_floor : log(res));
  }
}

double bob::ap::Ceps::logEnergy(blitz::Array<double,1> &data) const
{
  blitz::Array<double,1> data_p(data(blitz::Range(0,(int)m_win_length-1)));
  double gain = blitz::sum(blitz::pow2(data_p));
  return (gain < m_energy_floor ? m_log_energy_floor : log(gain)); 
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
  for(int l=1; l<=(int)m_delta_win; ++l) {
    blitz::Range rout(l,n_frames-l-1);
    blitz::Range rp(2*l,n_frames-1);
    blitz::Range rn(0,n_frames-2*l-1);
    output(rout,rall) += l*(input(rp,rall) - input(rn,rall));
  }

  const double factor = m_delta_win*(m_delta_win+1)/2;
  // Continue to fill the left boundary part as follows:
  // \f$output[i] += (\sum_{l=1+i}^{DW} l*input[i+l]) - (\sum_{l=i+1}^{DW}l)*input[0])\f$
  for(int i=0; i<(int)m_delta_win; ++i) {
    output(i,rall) -= (factor - i*(i+1)/2) * input(0,rall);
    for(int l=1+i; l<=(int)m_delta_win; ++l) {
      output(i,rall) += l*(input(i+l,rall));
    }
  }
  // Continue to fill the right boundary part as follows:
  // \f$output[i] += (\sum_{l=Nframes-1-i}^{DW}l)*input[Nframes-1]) - (\sum_{l=Nframes-1-i}^{DW} l*input[i-l])\f$
  for(int i=n_frames-(int)m_delta_win; i<n_frames;  ++i) {
    int ii = (n_frames-1)-i;
    output(i,rall) += (factor - ii*(ii+1)/2) * input(n_frames-1,rall);
    for(int l=1+ii; l<=(int)m_delta_win; ++l) {
      output(i,rall) -= l*input(i-l,rall);
    }
  }
  // Sum of the integer squared from 1 to delta_win
  const double sum = m_delta_win*(m_delta_win+1)*(2*m_delta_win+1)/3;
  output /= sum;
}

bob::ap::TestCeps::TestCeps(Ceps& ceps): m_ceps(ceps) {
}


