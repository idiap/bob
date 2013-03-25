/**
 * @file ip/cxx/GaborBankSpatial.cc
 * @date Wed Apr 13 20:45:32 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to perform a Gabor bank filtering
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

#include <bob/ip/GaborBankSpatial.h>
#include <bob/core/assert.h>

bob::ip::GaborBankSpatial::GaborBankSpatial( const int n_orient, const int n_freq,
  const double fmax, const bool orientation_full, const double k, 
  const double p, const double gamma, const double eta, 
  const int spatial_size, const bool cancel_dc, 
  const enum bob::ip::Gabor::NormOption norm_opt,
  //  const enum sp::Convolution::SizeOption size_opt,
  const enum sp::Extrapolation::BorderType border_type):
  m_n_orient(n_orient), m_n_freq(n_freq), m_fmax(fmax), 
  m_orientation_full(orientation_full), m_k(k), m_p(p), m_gamma(gamma), 
  m_eta(eta), m_spatial_size(spatial_size), m_cancel_dc(cancel_dc),
  m_norm_opt(norm_opt), // m_size_opt(size_opt), 
  m_border_type(border_type)
{
  computeFilters();
}

bob::ip::GaborBankSpatial::~GaborBankSpatial() { }

void bob::ip::GaborBankSpatial::operator()( 
  const blitz::Array<std::complex<double>,2>& src,
  blitz::Array<std::complex<double>,3>& dst)
{ 
  // Check input
  bob::core::array::assertZeroBase(src);

  // Check and resize dst if required 
  bob::core::array::assertZeroBase(dst);
  const blitz::TinyVector<int,3> shape(m_n_freq*m_n_orient, src.extent(0),
    src.extent(1));
  bob::core::array::assertSameShape(dst, shape);

  // Filter using the filter bank
  for( int i=0; i<m_n_freq*m_n_orient; ++i) {
    blitz::Array<std::complex<double>,2> dst_i = 
      dst( i, blitz::Range::all(), blitz::Range::all() );
    m_filters[i]->operator()(src, dst_i);
  }
}

void bob::ip::GaborBankSpatial::computeFreqs()
{
  m_freqs.resize(m_n_freq);
  m_freqs(0) = m_fmax;
  for(int i=1; i<m_n_freq; ++i)
    m_freqs(i) = m_freqs(i-1) / m_k;
}

void bob::ip::GaborBankSpatial::computeOrients()
{
  m_orients.resize(m_n_orient);
  for(int i=0; i<m_n_orient; ++i)
    m_orients(i) = ((m_orientation_full?2:1) * M_PI * i ) / m_n_orient;
}

void bob::ip::GaborBankSpatial::computeFilters()
{
  // Compute the set of frequencies and orientations
  computeFreqs();
  computeOrients();

  // Erase previous filters if any
  m_filters.clear();

  // Filter using the filter bank
  for( int i=0; i<m_n_freq*m_n_orient; ++i) {
    
    int f = i / m_n_orient;
    int o = i % m_n_orient;
    boost::shared_ptr<bob::ip::GaborSpatial> ptr( 
      new bob::ip::GaborSpatial( m_freqs(f), m_orients(o), m_gamma, m_eta, 
        m_spatial_size,  m_cancel_dc, m_norm_opt, /*m_size_opt,*/ 
        m_border_type) );
    m_filters.push_back( ptr);
  }
}

