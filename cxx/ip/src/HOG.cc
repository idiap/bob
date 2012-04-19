/**
 * @file cxx/ip/src/HOG.cc
 * @date Sat Apr 14 21:13:44 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "ip/HOG.h"
#include "ip/Exception.h"
#include "ip/block.h"
#include "core/array_assert.h"

namespace ip = bob::ip;

void ip::hogComputeCellHistogram(const blitz::Array<double,2>& mag, 
  const blitz::Array<double,2>& ori, blitz::Array<double,1>& hist, 
  const bool init_hist, const bool full_orientation)
{
  // Checks input arrays
  bob::core::array::assertSameShape(mag, ori);

  // Computes histogram
  hogComputeCellHistogram_(mag, ori, hist, init_hist, full_orientation);
}

void ip::hogComputeCellHistogram_(const blitz::Array<double,2>& mag, 
  const blitz::Array<double,2>& ori, blitz::Array<double,1>& hist, 
  const bool init_hist, const bool full_orientation)
{
  static const double range_orientation = (full_orientation? 2*M_PI : M_PI);
  const int nb_bins = hist.extent(0);

  // Initializes output to zero if required
  if(init_hist) hist = 0.;

  for(int i=0; i<mag.extent(0); ++i)
    for(int j=0; j<mag.extent(1); ++j)
    {
      double energy = mag(i,j);
      double orientation = ori(i,j);
      // Makes the orientation belongs to [0,2 PI] if required 
      // (because of atan2 implementation) 
      if(orientation < 0.) orientation += 2*M_PI; 
      // Makes the orientation belongs to [0, PI] if required 
      if( !full_orientation && (orientation > M_PI) ) orientation -= M_PI;

      // Computes "real" value of the closest bin
      double bin = orientation / range_orientation * nb_bins;
      // Computes the value of the "inferior" bin 
      // ("superior" bin corresponds to the one after the inferior bin)
      int bin_index = floor(bin);
      // Computes the weight for the "inferior" bin
      double weight = 1.-(bin-bin_index);

      // Updates the histogram (bilinearly)
      hist(bin_index % nb_bins) += weight * energy;
      hist((bin_index+1) % nb_bins) += (1. - weight) * energy;
    }
}

ip::HOGGradientMaps::HOGGradientMaps(const size_t height, 
  const size_t width, const hog::GradientMagnitudeType mag_type):
    m_gy(height, width), m_gx(height, width), m_mag_type(mag_type)
{
}

void ip::HOGGradientMaps::resize(const size_t height, const size_t width)
{
  m_gy.resize(height,width);
  m_gx.resize(height,width);
}

void ip::HOGGradientMaps::setHeight(const size_t height)
{
  m_gy.resize(height,m_gy.extent(1));
  m_gx.resize(height,m_gx.extent(1));
}

void ip::HOGGradientMaps::setWidth(const size_t width)
{
  m_gy.resize(m_gy.extent(0),width);
  m_gx.resize(m_gx.extent(0),width);
}

void ip::HOGGradientMaps::setGradientMagnitudeType(
  const hog::GradientMagnitudeType mag_type)
{
  m_mag_type = mag_type;
}

ip::HOG::HOG(const size_t height, const size_t width, 
    const size_t nb_bins, const bool full_orientation, const size_t cell_y,
    const size_t cell_x, const size_t cell_ov_y, const size_t cell_ov_x,
    const size_t block_y, const size_t block_x, const size_t block_ov_y, 
    const size_t block_ov_x):
  m_height(height), m_width(width),
  m_hog_gradient_maps(new ip::HOGGradientMaps(height, width, ip::hog::Magnitude)),
  m_nb_bins(nb_bins), m_full_orientation(full_orientation), 
  m_cell_y(cell_y), m_cell_x(cell_x), 
  m_cell_ov_y(cell_ov_y), m_cell_ov_x(cell_ov_x), 
  m_block_y(block_y), m_block_x(block_x), 
  m_block_ov_y(block_ov_y), m_block_ov_x(block_ov_x), 
  m_block_norm(ip::hog::L2), m_block_norm_eps(1e-10), m_block_norm_threshold(0.2) 
{
  resizeCache();
}

void ip::HOG::resize(const size_t height, const size_t width)
{
  m_height = height;
  m_width = width;
  resizeCache();
}

void ip::HOG::resizeCache()
{
  // Resize arrays for the Gradient maps
  m_magnitude.resize(m_height, m_width);
  m_orientation.resize(m_height, m_width);
  m_hog_gradient_maps->resize(m_height, m_width);
  // Resize everything else
  resizeCellCache();
}

void ip::HOG::resizeCellCache()
{
  // Resize the cell-related arrays
  const blitz::TinyVector<int,4> r = bob::ip::getBlock4DOutputShape(m_height, 
      m_width, m_cell_y, m_cell_x, m_cell_ov_y, m_cell_ov_x);
  m_cell_magnitude.resize(r(0), r(1), r(2), r(3));
  m_cell_orientation.resize(r(0), r(1), r(2), r(3));
  m_cell_hist.resize(r(0), r(1), m_nb_bins);
  
  // Number of blocks should be updated
  resizeBlockCache();
}

void ip::HOG::resizeBlockCache()
{
  // Determines the number of block per row and column
  blitz::TinyVector<int,4> nb_blocks = ip::getBlock4DOutputShape(
    m_cell_hist.extent(0), m_cell_hist.extent(1), m_block_y, m_block_x, 
    m_block_ov_y, m_block_ov_x);
  // Updates the class members
  m_nb_blocks_y = nb_blocks(0);
  m_nb_blocks_x = nb_blocks(1);
}

const blitz::TinyVector<int,3> ip::HOG::getOutputShape() const
{
  // Returns results
  blitz::TinyVector<int,3> res;
  res(0) = m_nb_blocks_y;
  res(1) = m_nb_blocks_x;
  res(2) = m_block_y * m_block_x * m_nb_bins;
  return res;
}

void ip::HOG::disableBlockNormalization()
{
  m_block_y = 1;
  m_block_x = 1;
  m_block_ov_y = 0;
  m_block_ov_x = 0;
  m_block_norm = ip::hog::None;
  resizeBlockCache();
}

