/**
 * @file cxx/ip/ip/block.h
 * @date Tue Apr 5 12:38:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to perform a decomposition by block.
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

#ifndef BOB_IP_BLOCK_H
#define BOB_IP_BLOCK_H

#include "core/array_assert.h"
#include "ip/Exception.h"
#include "ip/crop.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which performs a block decomposition of a 2D 
        *   blitz::array/image of a given type
        */
      template<typename T, typename U>
      void blockReferenceNoCheck(const blitz::Array<T,2>& src, U& dst, 
          const int block_h, const int block_w, const int overlap_h, 
          const int overlap_w)
      {
        // Determine the number of block per row and column
        const int size_ov_h = block_h - overlap_h;
        const int size_ov_w = block_w - overlap_w;
        const int n_blocks_h = (src.extent(0)-overlap_h)/ size_ov_h;
        const int n_blocks_w = (src.extent(1)-overlap_w)/ size_ov_w;

        // Perform the block decomposition
        blitz::Array<T,2> current_block;
        for( int h=0; h<n_blocks_h; ++h)
          for( int w=0; w<n_blocks_w; ++w) {
            detail::cropNoCheckReference( src, current_block, h*size_ov_h,
              w*size_ov_w, block_h, block_w);
            dst.push_back( current_block );
          }
      }

      /**
        * @brief Function which performs a block decomposition of a 2D 
        *   blitz::array/image of a given type
        */
      template<typename T>
      void blockNoCheck(const blitz::Array<T,2>& src, blitz::Array<T,3>& dst,
          const int block_h, const int block_w, const int overlap_h, 
          const int overlap_w)
      {
        // Determine the number of block per row and column
        const int size_ov_h = block_h - overlap_h;
        const int size_ov_w = block_w - overlap_w;
        const int n_blocks_h = (src.extent(0)-overlap_h)/ size_ov_h;
        const int n_blocks_w = (src.extent(1)-overlap_w)/ size_ov_w;

        // Perform the block decomposition
        blitz::Array<bool,2> src_mask, dst_mask;
        for( int h=0; h<n_blocks_h; ++h)
          for( int w=0; w<n_blocks_w; ++w) {
            blitz::Array<T,2> current_block = 
              dst( h*n_blocks_w+w, blitz::Range::all(), blitz::Range::all());
            cropNoCheck<T,false>( src, src_mask, current_block, dst_mask, 
              h*size_ov_h, w*size_ov_w, block_h, block_w, true);
          }
      }

      /**
        * @brief Function which checks the given parameters for a block 
        *   decomposition of a 2D blitz::array/image.
        */
      void blockCheckInput(const int height, const int width, 
          const int block_h, const int block_w, const int overlap_h, 
          const int overlap_w);

      /**
        * @brief Function which checks the given parameters for a block 
        *   decomposition of a 2D blitz::array/image.
        */
      template<typename T>
      void blockCheckInput(const blitz::Array<T,2>& src, const int block_h, 
          const int block_w, const int overlap_h, const int overlap_w)
      {
        // Checks that the src array has zero base indices
        bob::core::array::assertZeroBase( src);

        // Checks the parameters
        blockCheckInput(src.extent(0), src.extent(1), block_h, block_w,
          overlap_h, overlap_w);
      }

    }

    /**
      * @brief Function which performs a decomposition by block of a 2D 
      *   blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The returned blocks will refer to the same data as the in
      *   input 2D blitz array.
      * @param src The input blitz array
      * @param dst The STL container of 2D block blitz arrays. The STL 
      *   container requires to support the push_back method, such as
      *   a STL vector or list.
      * @param block_w The desired width of the blocks.
      * @param block_h The desired height of the blocks.
      * @param overlap_w The overlap between each block along the x axis.
      * @param overlap_h The overlap between each block along the y axis.
      */
    template<typename T, typename U>
    void blockReference(const blitz::Array<T,2>& src, U& dst, 
      const int block_h, const int block_w, const int overlap_h, 
      const int overlap_w)
    {
      // Check input
      detail::blockCheckInput( src, block_h, block_w, overlap_h, overlap_w);

      // Crop the 2D array
      detail::blockReferenceNoCheck(src, dst, block_h, block_w, overlap_h, overlap_w);
    }

    /**
      * @brief Function which returns the number of blocks when applying 
      *   a decomposition by block of a 2D blitz::array/image of a given size.
      * @param height  The height of the input array
      * @param width   The width of the input array
      * @param block_w The desired width of the blocks.
      * @param block_h The desired height of the blocks.
      * @param overlap_w The overlap between each block along the x axis.
      * @param overlap_h The overlap between each block along the y axis.
      */ 
    const blitz::TinyVector<int,3> getBlockShape(const int height, 
      const int width, const int block_h, const int block_w, 
      const int overlap_h, const int overlap_w);

    /**
      * @brief Function which returns the number of blocks when applying 
      *   a decomposition by block of a 2D blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param block_w The desired width of the blocks.
      * @param block_h The desired height of the blocks.
      * @param overlap_w The overlap between each block along the x axis.
      * @param overlap_h The overlap between each block along the y axis.
      */
    template<typename T>
    const blitz::TinyVector<int,3> getBlockShape(const blitz::Array<T,2>& src,
      const int block_h, const int block_w, const int overlap_h, 
      const int overlap_w)
    {
      // Check input
      detail::blockCheckInput( src, block_h, block_w, overlap_h, overlap_w);

      // Check paramters and returns result
      return getBlockShape(src.extent(0), src.extent(1), block_h, block_w, 
        overlap_h, overlap_w);
    }

    /**
      * @brief Function which performs a decomposition by block of a 2D 
      *   blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param dst The STL container of 2D block blitz arrays. The STL 
      *   container requires to support the push_back method, such as
      *   a STL vector or list.
      * @param block_w The desired width of the blocks.
      * @param block_h The desired height of the blocks.
      * @param overlap_w The overlap between each block along the x axis.
      * @param overlap_h The overlap between each block along the y axis.
      */
    template<typename T>
    void block(const blitz::Array<T,2>& src, blitz::Array<T,3>& dst, 
      const int block_h, const int block_w, const int overlap_h, 
      const int overlap_w)
    {
      // Check input
      detail::blockCheckInput( src, block_h, block_w, overlap_h, overlap_w);
      blitz::TinyVector<int,3> shape = 
        getBlockShape(src, block_h, block_w, overlap_h, overlap_w);
      bob::core::array::assertSameShape( dst, shape);

      // Crop the 2D array
      detail::blockNoCheck(src, dst, block_h, block_w, overlap_h, overlap_w);
    }

    /**
      * @brief Function which returns the number of blocks along y and x when
      *   applying a decomposition by block of an input array of the given size.
      * @param height  The height of the input array
      * @param width   The width of the input array
      * @param block_w The desired width of the blocks.
      * @param block_h The desired height of the blocks.
      * @param overlap_w The overlap between each block along the x axis.
      * @param overlap_h The overlap between each block along the y axis.
      * @return Number of blocks along y and x-axis
      */
    const blitz::TinyVector<int,2> getNBlocks(const int height, 
      const int width, const int block_h, const int block_w, 
      const int overlap_h, const int overlap_w);

    /**
      * @brief Function which returns the number of blocks along y and x when
      *   applying a decomposition by block of a 2D blitz::array/image.
      *   The first dimension is the number of blocks along y-axis, whereas the second
      *   one is the nomber of blocks along x-axis.
      * @param src The input blitz array
      * @param block_w The desired width of the blocks.
      * @param block_h The desired height of the blocks.
      * @param overlap_w The overlap between each block along the x axis.
      * @param overlap_h The overlap between each block along the y axis.
      * @return Number of blocks along y and x-axis
      */
    template<typename T>
    const blitz::TinyVector<int,2> getNBlocks(const blitz::Array<T,2>& src,
                                              const int block_h, const int block_w,
                                              const int overlap_h, const int overlap_w)
    {
      // Check input
      detail::blockCheckInput( src, block_h, block_w, overlap_h, overlap_w);

      // Check paramters and returns result
      return getNBlocks(src.extent(0), src.extent(1), block_h, block_w, 
        overlap_h, overlap_w);
    }
    
  }

/**
 * @}
 */
}

#endif /* BOB_IP_BLOCK_H */
