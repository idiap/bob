/**
 * @file ip/cxx/VLDSIFT.cc
 * @date Mon Jan 23 20:46:07 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Dense SIFT implementation using VLFeat
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

#include "bob/ip/VLDSIFT.h"

#include "bob/core/array_assert.h"
#include "bob/core/check.h"
#include "bob/core/array_copy.h"

bob::ip::VLDSIFT::VLDSIFT(const size_t height, const size_t width, 
  const size_t step, const size_t block_size):
    m_height(height), m_width(width), m_step_y(step), m_step_x(step),
    m_block_size_y(block_size), m_block_size_x(block_size)
{
  allocateAndInit();
}


bob::ip::VLDSIFT::VLDSIFT(const VLDSIFT& other):
  m_height(other.m_height), m_width(other.m_width), 
  m_step_y(other.m_step_y), m_step_x(other.m_step_x),
  m_block_size_y(other.m_block_size_y), 
  m_block_size_x(other.m_block_size_x), 
  m_use_flat_window(other.m_use_flat_window),
  m_window_size(other.m_window_size)
{
  allocateAndSet();
}

bob::ip::VLDSIFT::~VLDSIFT()
{
  cleanup();
}

bob::ip::VLDSIFT& bob::ip::VLDSIFT::operator=(const bob::ip::VLDSIFT& other)
{
  if (this != &other)
  {
    m_height = other.m_height;
    m_width = other.m_width;
    m_step_y = other.m_step_y;
    m_step_x = other.m_step_x;
    m_block_size_y = other.m_block_size_y;
    m_block_size_x = other.m_block_size_x;
    m_use_flat_window = other.m_use_flat_window;
    m_window_size = other.m_window_size;
  
    // Allocates filter, and set filter properties
    allocateAndSet();
  }
  return *this;
}

bool bob::ip::VLDSIFT::operator==(const bob::ip::VLDSIFT& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width && 
          this->m_step_y == b.m_step_y && this->m_step_x == b.m_step_x &&
          this->m_block_size_y == b.m_block_size_y &&
          this->m_block_size_x == b.m_block_size_x &&
          this->m_use_flat_window == b.m_use_flat_window &&
          this->m_window_size == b.m_window_size); 
}

bool bob::ip::VLDSIFT::operator!=(const bob::ip::VLDSIFT& b) const
{
  return !(this->operator==(b));
}

void bob::ip::VLDSIFT::setBlockSizeY(const size_t block_size_y)
{ 
  m_block_size_y = block_size_y;
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(m_filt);
  geom.binSizeY = (int)m_block_size_y;
  geom.binSizeX = (int)m_block_size_x;
  vl_dsift_set_geometry(m_filt, &geom) ;
}

void bob::ip::VLDSIFT::setBlockSizeX(const size_t block_size_x)
{ 
  m_block_size_x = block_size_x;
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(m_filt);
  geom.binSizeY = (int)m_block_size_y;
  geom.binSizeX = (int)m_block_size_x;
  vl_dsift_set_geometry(m_filt, &geom) ;
}

void bob::ip::VLDSIFT::operator()(const blitz::Array<float,2>& src, 
  blitz::Array<float,2>& dst)
{
  // Check parameters size size
  bob::core::array::assertSameDimensionLength(src.extent(0), m_height);
  bob::core::array::assertSameDimensionLength(src.extent(1), m_width);
  int num_frames = vl_dsift_get_keypoint_num(m_filt);
  int descr_size = vl_dsift_get_descriptor_size(m_filt);
  bob::core::array::assertSameDimensionLength(dst.extent(0), num_frames);
  bob::core::array::assertSameDimensionLength(dst.extent(1), descr_size);

  // Get C-style pointer to src data, making a copy if required
  const float* data;
  blitz::Array<float,2> x;
  if(bob::core::array::isCZeroBaseContiguous(src))
    data = src.data();
  else
  {
    x.reference(bob::core::array::ccopy(src));
    data = x.data();
  }
 
  // Computes features
  vl_dsift_process(m_filt, data);

  // Move output back to destination array
  float const *descrs = vl_dsift_get_descriptors(m_filt);
  if(bob::core::array::isCZeroBaseContiguous(dst)) 
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

void bob::ip::VLDSIFT::allocate()
{
  // Generates the filter
  m_filt = vl_dsift_new_basic((int)m_width, (int)m_height, (int)m_step_y, 
            (int)m_block_size_y);
}

void bob::ip::VLDSIFT::allocateAndInit()
{
  allocate();
  m_use_flat_window = vl_dsift_get_flat_window(m_filt);
  m_window_size = vl_dsift_get_window_size(m_filt);
}

void bob::ip::VLDSIFT::setFilterProperties()
{
  // Set filter properties
  vl_dsift_set_steps(m_filt, (int)m_step_x, (int)m_step_y);
  vl_dsift_set_flat_window(m_filt, m_use_flat_window);
  vl_dsift_set_window_size(m_filt, m_window_size);
  // Set block size
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(m_filt);
  geom.binSizeY = (int)m_block_size_y;
  geom.binSizeX = (int)m_block_size_x;
  vl_dsift_set_geometry(m_filt, &geom) ;
}

void bob::ip::VLDSIFT::allocateAndSet()
{
  allocate();
  setFilterProperties();
}

void bob::ip::VLDSIFT::cleanup()
{
  // Releases filter
  vl_dsift_delete(m_filt);
  m_filt = 0;
}

