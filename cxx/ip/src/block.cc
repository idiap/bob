/**
 * @file cxx/ip/src/block.cc
 * @date Mon Apr 16 18:03:44 2012 +0200
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

#include "ip/block.h"
#include "ip/Exception.h"

namespace ip = bob::ip;
namespace ipd = bob::ip::detail;

/**
  * @brief Function which checks the given parameters for a block 
  *   decomposition of a 2D blitz::array/image.
  */
void ipd::blockCheckInput(const int height, const int width, const int block_h, 
    const int block_w, const int overlap_h, const int overlap_w)
{
  // Check parameters and throw exception if required
  if( block_h<1)
    throw ParamOutOfBoundaryError("block_h", false, block_h, 1); 
  if( block_h>height )
    throw ParamOutOfBoundaryError("block_h", true, block_h, 
      height); 
  if( block_w<1)
    throw ParamOutOfBoundaryError("block_w", false, block_w, 1); 
  if( block_w>width )
    throw ParamOutOfBoundaryError("block_w", true, block_w, 
      width); 
  if( overlap_h<0)
    throw ParamOutOfBoundaryError("overlap_h", false, overlap_h, 1);
  if( overlap_h>=block_h )
    throw ParamOutOfBoundaryError("overlap_h", true, overlap_h, 
      block_h); 
  if( overlap_w<0)
    throw ParamOutOfBoundaryError("overlap_w", false, overlap_w, 1);
  if( overlap_w>=block_w )
    throw ParamOutOfBoundaryError("overlap_w", true, overlap_w, 
      block_w); 
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
const blitz::TinyVector<int,3> ip::getBlock3DOutputShape(const int height, 
  const int width, const int block_h, const int block_w, const int overlap_h,
  const int overlap_w)
{
  // Determine the number of block per row and column
  const int size_ov_h = block_h - overlap_h;
  const int size_ov_w = block_w - overlap_w;
  const int n_blocks_h = (height-overlap_h)/ size_ov_h;
  const int n_blocks_w = (width-overlap_w)/ size_ov_w;

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
const blitz::TinyVector<int,4> ip::getBlock4DOutputShape(const int height, 
  const int width, const int block_h, const int block_w, const int overlap_h,
  const int overlap_w)
{
  // Determine the number of block per row and column
  const int size_ov_h = block_h - overlap_h;
  const int size_ov_w = block_w - overlap_w;
  const int n_blocks_h = (height-overlap_h)/ size_ov_h;
  const int n_blocks_w = (width-overlap_w)/ size_ov_w;

  // Return the shape of the output
  blitz::TinyVector<int,4> res( n_blocks_h, n_blocks_w, block_h, block_w);
  return res;
}

