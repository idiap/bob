/**
 * @file ip/cxx/block.cc
 * @date Mon Apr 16 18:03:44 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <stdexcept>
#include <boost/format.hpp>
#include "bob/ip/block.h"

/**
  * @brief Function which checks the given parameters for a block
  *   decomposition of a 2D blitz::array/image.
  */
void bob::ip::detail::blockCheckInput(const size_t height,
  const size_t width, const size_t block_h, const size_t block_w,
  const size_t overlap_h, const size_t overlap_w)
{
  // Check parameters and throw exception if required
  if (block_h < 1 || block_h > height) {
    boost::format m("setting `block_h' to %lu is outside the expected range [1, %lu]");
    m % block_h % height;
    throw std::runtime_error(m.str());
  }
  if (block_w < 1 || block_h > width) {
    boost::format m("setting `block_w' to %lu is outside the expected range [1, %lu]");
    m % block_w % width;
    throw std::runtime_error(m.str());
  }
  if (overlap_h >= block_h) {
    boost::format m("setting `overlap_h' to %lu is outside the expected range [0, %lu]");
    m % overlap_h % (block_h-1);
    throw std::runtime_error(m.str());
  }
  if (overlap_w >= block_w) {
    boost::format m("setting `overlap_w' to %lu is outside the expected range [0, %lu]");
    m % overlap_w % (block_w-1);
    throw std::runtime_error(m.str());
  }
}

/**
  * @brief Function which returns the number of blocks when applying
  *   a decomposition by block of an input array of the given size.
  * @param height  The height of the input array
  * @param width   The width of the input array
  * @param block_h The desired height of the blocks.
  * @param block_w The desired width of the blocks.
  * @param overlap_h The overlap between each block along the y axis.
  * @param overlap_w The overlap between each block along the x axis.
  */
const blitz::TinyVector<int,3>
bob::ip::getBlock3DOutputShape(const size_t height, const size_t width,
  const size_t block_h, const size_t block_w,
  const size_t overlap_h, const size_t overlap_w)
{
  // Determine the number of block per row and column
  const int size_ov_h = block_h - overlap_h;
  const int size_ov_w = block_w - overlap_w;
  const int n_blocks_h = (int)(height-overlap_h) / size_ov_h;
  const int n_blocks_w = (int)(width-overlap_w) / size_ov_w;

  // Return the shape of the output
  blitz::TinyVector<int,3> res( n_blocks_h*n_blocks_w, block_h, block_w);
  return res;
}


/**
  * @brief Function which returns the number of blocks when applying
  *   a decomposition by block of an input array of the given size.
  * @param height  The height of the input array
  * @param width   The width of the input array
  * @param block_h The desired height of the blocks.
  * @param block_w The desired width of the blocks.
  * @param overlap_h The overlap between each block along the y axis.
  * @param overlap_w The overlap between each block along the x axis.
  */
const blitz::TinyVector<int,4>
bob::ip::getBlock4DOutputShape(const size_t height, const size_t width,
  const size_t block_h, const size_t block_w,
  const size_t overlap_h, const size_t overlap_w)
{
  // Determine the number of block per row and column
  const int size_ov_h = block_h - overlap_h;
  const int size_ov_w = block_w - overlap_w;
  const int n_blocks_h = (int)(height-overlap_h) / size_ov_h;
  const int n_blocks_w = (int)(width-overlap_w) / size_ov_w;

  // Return the shape of the output
  blitz::TinyVector<int,4> res( n_blocks_h, n_blocks_w, block_h, block_w);
  return res;
}

