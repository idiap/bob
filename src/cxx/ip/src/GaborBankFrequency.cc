/**
 * @file src/cxx/ip/src/GaborBankFrequency.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to perform a Gabor bank filtering
 */

#include "ip/GaborBankFrequency.h"
#include "core/array_assert.h"
#include "core/array_copy.h"

#include <iostream>

namespace tca = Torch::core::array;
namespace ip = Torch::ip;

ip::GaborBankFrequency::GaborBankFrequency( const int height, const int width,
  const int n_orient, const int n_freq, const double fmax, 
  const bool orientation_full, const double k, 
  const double p, const bool optimal_gamma_eta, 
  const double gamma, const double eta, 
  const double pf, const bool cancel_dc, 
  const bool use_envelope, const bool output_in_frequency):
  m_height(height), m_width(width), m_n_orient(n_orient), m_n_freq(n_freq), 
  m_fmax(fmax), m_orientation_full(orientation_full), m_k(k), m_p(p), 
  m_optimal_gamma_eta(optimal_gamma_eta),
  m_gamma(gamma), m_eta(eta), m_pf(pf), m_cancel_dc(cancel_dc),
  m_use_envelope(use_envelope), m_output_in_frequency(output_in_frequency),
  m_freqs(n_freq), m_orients(n_orient)
{
  computeFilters();
}

ip::GaborBankFrequency::GaborBankFrequency(const GaborBankFrequency& other):
  m_height(other.m_height), m_width(other.m_width), m_n_orient(other.m_n_orient), m_n_freq(other.m_n_freq),
  m_fmax(other.m_fmax), m_orientation_full(other.m_orientation_full), m_k(other.m_k), m_p(other.m_p), 
  m_optimal_gamma_eta(other.m_optimal_gamma_eta), m_gamma(other.m_gamma), m_eta(other.m_eta), 
  m_pf(other.m_pf), m_cancel_dc(other.m_cancel_dc), m_use_envelope(other.m_use_envelope), 
  m_output_in_frequency(other.m_output_in_frequency),
  m_freqs(Torch::core::array::ccopy(other.m_freqs)),
  m_orients(Torch::core::array::ccopy(other.m_orients))
{
  for(size_t i=0; i<other.m_filters.size(); ++i) {
    boost::shared_ptr<ip::GaborFrequency> ptr(new ip::GaborFrequency(*(other.m_filters[i])) );
    m_filters.push_back(ptr);
  }
}

ip::GaborBankFrequency::~GaborBankFrequency() {
}

void ip::GaborBankFrequency::operator()( 
  const blitz::Array<std::complex<double>,2>& src,
  blitz::Array<std::complex<double>,3>& dst)
{ 
  // Checks input
  tca::assertZeroBase(src);

  // Checks output
  tca::assertZeroBase(dst);
  const blitz::TinyVector<int,3> shape(m_n_freq*m_n_orient, src.extent(0),
    src.extent(1));
  tca::assertSameShape(dst, shape);

  // Filters using the filter bank
  for( int i=0; i<m_n_freq*m_n_orient; ++i) {
    blitz::Array<std::complex<double>,2> dst_i = 
      dst( i, blitz::Range::all(), blitz::Range::all() );
    m_filters[i]->operator()(src, dst_i);
  }
}

void ip::GaborBankFrequency::computeFreqs()
{
  m_freqs.resize(m_n_freq);
  m_freqs(0) = m_fmax;
  for(int i=1; i<m_n_freq; ++i)
    m_freqs(i) = m_freqs(i-1) / m_k;
}

void ip::GaborBankFrequency::computeOrients()
{
  m_orients.resize(m_n_orient);
  for(int i=0; i<m_n_orient; ++i)
    m_orients(i) = ((m_orientation_full?2:1) * M_PI * i ) / m_n_orient;
}

void ip::GaborBankFrequency::computeFilters()
{
  // Computes the set of frequencies and orientations
  computeFreqs();
  computeOrients();

  // Computes eta and gamma if required
  if(m_optimal_gamma_eta)
    computeOptimalGammaEta();

  // Erases previous filters if any
  m_filters.clear();

  // Filters using the filter bank
  for( int i=0; i<m_n_freq*m_n_orient; ++i) {
    
    int f = i / m_n_orient;
    int o = i % m_n_orient;
    boost::shared_ptr<ip::GaborFrequency> ptr(
      new ip::GaborFrequency( m_height, m_width, m_freqs(f), m_orients(o), 
        m_gamma, m_eta, m_pf,  m_cancel_dc, m_use_envelope, 
        m_output_in_frequency) );
    m_filters.push_back( ptr);
  }
}

void ip::GaborBankFrequency::computeOptimalGammaEta()
{
  // Computes and sets gamma and eta
  m_gamma =  ((m_k+1)/(m_k-1)) * sqrt(-log(m_p)) / M_PI;
  m_eta = 1. / ( tan( M_PI / ((m_orientation_full?1:2)*m_n_orient) ) *
                 sqrt(M_PI*M_PI/log(1./m_p) - 1./(m_gamma*m_gamma)) );
}
