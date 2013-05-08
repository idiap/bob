/**
 * @file ap/cxx/Spectrogram.cc
 * @date Wed Jan 11:09:30 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
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

#include <bob/ap/Spectrogram.h>
#include <bob/core/check.h>
#include <bob/core/assert.h>
#include <bob/core/cast.h>

bob::ap::Spectrogram::Spectrogram(const double sampling_frequency,
    const double win_length_ms, const double win_shift_ms,
    const size_t n_filters, const double f_min, const double f_max,
    const double pre_emphasis_coeff, const bool mel_scale):
  bob::ap::Energy(sampling_frequency, win_length_ms, win_shift_ms),
  m_n_filters(n_filters), m_f_min(f_min), m_f_max(f_max), 
  m_pre_emphasis_coeff(pre_emphasis_coeff), m_mel_scale(mel_scale),
  m_fb_out_floor(1.), m_energy_filter(false), m_log_filter(true),
  m_fft(1)
{
  // Check pre-emphasis coefficient
  if (pre_emphasis_coeff < 0. || pre_emphasis_coeff > 1.)
    throw bob::core::InvalidArgumentException("pre_emphasis_coeff", 
      pre_emphasis_coeff, 0., 1.);

  // Initialization
  initWinLength();
  initWinShift();

  // Initializes logarithm of flooring values
  m_log_fb_out_floor = log(m_fb_out_floor);

  m_cache_filters.resize(m_n_filters);
}

bob::ap::Spectrogram::Spectrogram(const Spectrogram& other):
  bob::ap::Energy(other), m_n_filters(other.m_n_filters), 
  m_f_min(other.m_f_min), m_f_max(other.m_f_max),
  m_pre_emphasis_coeff(other.m_pre_emphasis_coeff),
  m_mel_scale(other.m_mel_scale), m_fb_out_floor(other.m_fb_out_floor),
  m_energy_filter(other.m_energy_filter), m_log_filter(other.m_log_filter),
  m_fft(other.m_fft)
{
  // Initialization
  initWinLength();
  initWinShift();

  // Initializes logarithm of flooring values
  m_log_fb_out_floor = log(m_fb_out_floor);

  m_cache_filters.resize(m_n_filters);
}

bob::ap::Spectrogram& bob::ap::Spectrogram::operator=(const bob::ap::Spectrogram& other)
{
  if (this != &other)
  {
    bob::ap::Energy::operator=(other);
    m_n_filters = other.m_n_filters;
    m_f_min = other.m_f_min;
    m_f_max = other.m_f_max;
    m_pre_emphasis_coeff = other.m_pre_emphasis_coeff;
    m_mel_scale = other.m_mel_scale;
    m_fb_out_floor = other.m_fb_out_floor;
    m_energy_filter = other.m_energy_filter;
    m_log_filter = other.m_log_filter;
    m_fft = other.m_fft;

    // Initialization
    initWinLength();
    initWinShift();

    // Initializes logarithm of flooring values
    m_log_fb_out_floor = log(m_fb_out_floor);

    m_cache_filters.resize(m_n_filters);
  }
  return *this;
}

bool bob::ap::Spectrogram::operator==(const bob::ap::Spectrogram& other) const
{
  return (bob::ap::Energy::operator==(other) && 
          m_n_filters == other.m_n_filters && m_f_min == other.m_f_min &&
          m_f_max == other.m_f_max && 
          m_pre_emphasis_coeff == other.m_pre_emphasis_coeff &&
          m_mel_scale == other.m_mel_scale &&
          m_fb_out_floor == other.m_fb_out_floor &&
          m_energy_filter == other.m_energy_filter &&
          m_log_filter == other.m_log_filter); 
}

bool bob::ap::Spectrogram::operator!=(const bob::ap::Spectrogram& other) const
{
  return !(this->operator==(other));
}

bob::ap::Spectrogram::~Spectrogram()
{
}

blitz::TinyVector<int,2> 
bob::ap::Spectrogram::getShape(const size_t input_size) const
{
  // Res will contain the number of frames x the dimension of the feature vector
  blitz::TinyVector<int,2> res;

  // 1. Number of frames
  res(0) = 1+((input_size-m_win_length)/m_win_shift);

  // 2. Dimension of the feature vector
  res(1) = m_win_length;

  return res;
}

blitz::TinyVector<int,2>
bob::ap::Spectrogram::getShape(const blitz::Array<double,1>& input) const
{
  return getShape(input.extent(0));
}

void bob::ap::Spectrogram::setSamplingFrequency(const double sampling_frequency)
{
  bob::ap::Energy::setSamplingFrequency(sampling_frequency);
  initWinLength();
  initWinShift();
}

void bob::ap::Spectrogram::setWinLengthMs(const double win_length_ms)
{
  bob::ap::Energy::setWinLengthMs(win_length_ms);
  initWinLength();
}

void bob::ap::Spectrogram::setWinShiftMs(const double win_shift_ms)
{
  bob::ap::Energy::setWinShiftMs(win_shift_ms);
  initWinShift();
}

void bob::ap::Spectrogram::setNFilters(size_t n_filters)
{ 
  m_n_filters = n_filters; 
  m_cache_filters.resize(m_n_filters); 
  initCacheFilterBank(); 
}

void bob::ap::Spectrogram::setFMin(double f_min)
{ 
  m_f_min = f_min; 
  initCacheFilterBank(); 
}

void bob::ap::Spectrogram::setFMax(double f_max)
{ 
  m_f_max = f_max; 
  initCacheFilterBank();
}

void bob::ap::Spectrogram::setMelScale(bool mel_scale)
{ 
  m_mel_scale = mel_scale;
  initCacheFilterBank(); 
}

double bob::ap::Spectrogram::herzToMel(double f)
{
  return (2595.*log10(1+f/700.));
}

double bob::ap::Spectrogram::melToHerz(double f)
{
  return ((double)(700.*(pow(10,f/2595.)-1)));
}

void bob::ap::Spectrogram::initCacheHammingKernel()
{
  // Hamming Window initialization
  m_hamming_kernel.resize(m_win_length);
  double cst = 2*M_PI/(double)(m_win_length-1);
  blitz::firstIndex i;
  m_hamming_kernel = 0.54-0.46*blitz::cos(i*cst);
}

void bob::ap::Spectrogram::initCacheFilterBank()
{
  initCachePIndex();
  initCacheFilters();
}

void bob::ap::Spectrogram::initCachePIndex()
{
  // Computes the indices for the triangular filter bank
  m_p_index.resize(m_n_filters+2);
  // 'Mel' frequency decomposition (for MFCC)
  if (m_mel_scale)
  {
    double m_max = herzToMel(m_f_max);
    double m_min = herzToMel(m_f_min);
    for (int i=0; i<(int)m_n_filters+2; ++i) {
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
    for (int i=0; i<(int)m_n_filters+2; ++i) {
      m_p_index(i) = (int)round(cst_a * i + cst_b);
    }
  }
}

void bob::ap::Spectrogram::initCacheFilters()
{
  // Creates the Triangular filter bank
  m_filter_bank.clear();
  blitz::firstIndex ii;
  for (int i=0; i<(int)m_n_filters; ++i)
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

void bob::ap::Spectrogram::initWinLength()
{ 
  bob::ap::Energy::initWinLength();
  initCacheHammingKernel(); 
  initCacheFilterBank(); 
}

void bob::ap::Spectrogram::initWinSize()
{
  bob::ap::Energy::initWinSize();
  m_fft.reset(m_win_size);
  m_cache_frame_c1.resize(m_win_size);
  m_cache_frame_c2.resize(m_win_size);
}

void bob::ap::Spectrogram::pre_emphasis(blitz::Array<double,1> &data) const
{
  if (m_pre_emphasis_coeff != 0.)
  { 
    // Pre-emphasise the signal by applying the first order equation
    // \f$data_{n} := data_{n} − a*data_{n−1}\f$
    blitz::Range r0((int)m_win_length-2,0,-1); 
    blitz::Range r1((int)m_win_length-1,1,-1); 
    data(r1) -= m_pre_emphasis_coeff * data(r0); // Apply first order equation
    data(0) *= 1. - m_pre_emphasis_coeff; // Update first element
  }
}

void bob::ap::Spectrogram::hammingWindow(blitz::Array<double,1> &data) const
{
  blitz::Range r(0,(int)m_win_length-1);
  data(r) *= m_hamming_kernel;
}

void bob::ap::Spectrogram::filterBank(blitz::Array<double,1>& x)
{
  // Apply the FFT
  m_cache_frame_c1 = bob::core::array::cast<std::complex<double> >(x);
  m_fft(m_cache_frame_c1, m_cache_frame_c2);

  // Take the the power spectrum of the first part of the output of the FFT
  blitz::Range r(0,(int)m_win_size/2);
  blitz::Array<double,1> x_half(x(r));
  blitz::Array<std::complex<double>,1> complex_half(m_cache_frame_c2(r));
  x_half = blitz::abs(complex_half);
  if (m_energy_filter) // Apply the filter bank to the energy
    x_half = blitz::pow2(x_half);

  if (m_log_filter) // Apply the log triangular filter bank
    logTriangularFilterBank(x);
  else // Apply the triangular filter ban
    triangularFilterBank(x);
}

void bob::ap::Spectrogram::logTriangularFilterBank(blitz::Array<double,1>& data) const
{
  for (int i=0; i<(int)m_n_filters; ++i)
  {
    blitz::Array<double,1> data_slice(data(blitz::Range(m_p_index(i),m_p_index(i+2))));
    double res = blitz::sum(data_slice * m_filter_bank[i]);
    m_cache_filters(i)= (res < m_fb_out_floor ? m_log_fb_out_floor : log(res));
  }
}

void bob::ap::Spectrogram::triangularFilterBank(blitz::Array<double,1>& data) const
{
  for (int i=0; i<(int)m_n_filters; ++i)
  {
    blitz::Array<double,1> data_slice(data(blitz::Range(m_p_index(i),m_p_index(i+2))));
    m_cache_filters(i) = blitz::sum(data_slice * m_filter_bank[i]);
  }
}

void bob::ap::Spectrogram::operator()(const blitz::Array<double,1>& input,
  blitz::Array<double,2>& spectrogram_matrix)
{
  // Get expected dimensionality of output array
  blitz::TinyVector<int,2> spectrogram_shape = bob::ap::Spectrogram::getShape(input);
  // Check dimensionality of output array
  bob::core::array::assertSameShape(spectrogram_matrix, spectrogram_shape);
  int n_frames=spectrogram_shape(0);

  // Computes the center of the cut-off frequencies
  blitz::Range r1(0,m_win_size/2);
  for (int i=0; i<n_frames; ++i)
  {
    // Extract and normalize frame
    extractNormalizeFrame(input, i, m_cache_frame_d);

    // Apply pre-emphasis
    pre_emphasis(m_cache_frame_d);
    // Apply the Hamming window
    hammingWindow(m_cache_frame_d);
    // Filter with the triangular filter bank (either in linear or Mel domain)
    filterBank(m_cache_frame_d);

    blitz::Array<double,1> spec_matrix_row(spectrogram_matrix(i,r1));
    spec_matrix_row = m_cache_frame_d(r1).copy();
  }
}


