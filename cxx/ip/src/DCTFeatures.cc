/**
 * @file cxx/ip/src/DCTFeatures.cc
 * @date Mon Aug 27 20:29:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to DCT by blocks features
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

#include "ip/DCTFeatures.h"

bob::ip::DCTFeatures& 
bob::ip::DCTFeatures::operator=(const bob::ip::DCTFeatures& other)
{
  if (this != &other)
  {
    m_block_h = other.m_block_h;
    m_block_w = other.m_block_w;
    m_overlap_h = other.m_overlap_h;
    m_overlap_w = other.m_overlap_w;
    m_n_dct_coefs = other.m_n_dct_coefs;
    m_dct2d->reset(m_block_h, m_block_w);
  }
  return *this;
}

bool 
bob::ip::DCTFeatures::operator==(const bob::ip::DCTFeatures& b) const
{
  return (this->m_block_h == b.m_block_h && this->m_block_w == b.m_block_w && 
          this->m_overlap_h == b.m_overlap_h && 
          this->m_overlap_w == b.m_overlap_w && 
          this->m_n_dct_coefs == b.m_n_dct_coefs);
}

bool 
bob::ip::DCTFeatures::operator!=(const bob::ip::DCTFeatures& b) const
{
  return !(this->operator==(b));
}

template <> 
void bob::ip::DCTFeatures::operator()<double>(const blitz::Array<double,2>& src, 
  blitz::Array<double, 2>& dst) const
{ 
  // Checks input/output
  bob::core::array::assertZeroBase(src);
  bob::core::array::assertZeroBase(dst);
  blitz::TinyVector<int,2> shape(getNBlocks(src), m_n_dct_coefs);
  bob::core::array::assertSameShape(dst, shape);
 
  // get all the blocks
  std::list<blitz::Array<double,2> > blocks;
  blockReference(src, blocks, m_block_h, m_block_w, m_overlap_h, m_overlap_w);
 
  /// dct extract each block
  int i=0;
  for(std::list<blitz::Array<double,2> >::const_iterator it = blocks.begin();
    it != blocks.end(); ++it, ++i) 
  {
    // Extract DCT for the current block
    // TODO: avoid the copy if possible
    m_dct2d->operator()(bob::core::array::ccopy(*it), m_cache_block);

    // Extract the required number of coefficients using the zigzag pattern
    // and push it in the right dst row
    blitz::Array<double,1> dst_row = dst(i, blitz::Range::all());
    zigzag(m_cache_block, dst_row);
  }
}

