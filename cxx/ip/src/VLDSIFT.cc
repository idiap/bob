/**
 * @file cxx/ip/src/VLDSIFT.cc
 * @date Mon Jan 23 20:46:07 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Dense SIFT implementation using VLFeat
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

#include "ip/VLDSIFT.h"

#include "core/array_assert.h"
#include "core/array_check.h"
#include "core/array_copy.h"

#include "core/logging.h"

namespace ip = bob::ip;
namespace ca = bob::core::array;

ip::VLDSIFT::VLDSIFT(const int height, const int width, const int step,
    const int block_size):
  m_height(height), m_width(width), m_step(step), m_block_size(block_size)
{
  m_filt = vl_dsift_new_basic(m_width, m_height, step, block_size);
}

void ip::VLDSIFT::operator()(const blitz::Array<float,2>& src, 
  blitz::Array<float,2>& dst)
{
  // Check parameters size size
  ca::assertSameDimensionLength(src.extent(0), m_height);
  ca::assertSameDimensionLength(src.extent(1), m_width);
  int num_frames = vl_dsift_get_keypoint_num(m_filt);
  int descr_size = vl_dsift_get_descriptor_size(m_filt);
  ca::assertSameDimensionLength(dst.extent(0), num_frames);
  ca::assertSameDimensionLength(dst.extent(1), descr_size);

  // Get C-style pointer to src data, making a copy if required
  const float* data;
  blitz::Array<float,2> x;
  if(ca::isCZeroBaseContiguous(src))
    data = src.data();
  else
  {
    x.reference(ca::ccopy(src));
    data = x.data();
  }
 
  // Computes features
  vl_dsift_process(m_filt, data);

  // Move output back to destination array
  float const *descrs = vl_dsift_get_descriptors(m_filt);
  if(ca::isCZeroBaseContiguous(dst)) 
    // fast copy
    memcpy(dst.data(), descrs, num_frames*descr_size);
  else
  {
    // Iterate (slow...)
    for(int f=0; f<num_frames; ++f)
      for(int b=0; b<descr_size; ++b) 
      {
        dst(f,b) = *descrs;
        ++descrs;
      }
  }
}



ip::VLDSIFT::~VLDSIFT()
{
  // Releases filter
  vl_dsift_delete(m_filt);
}
