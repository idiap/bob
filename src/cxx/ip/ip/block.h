/**
 * @file src/cxx/ip/ip/block.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:Niklas.Johansson@idiap.ch">Niklas Johansson</a> 
 *
 * @brief This file defines a function to perform a decomposition by block.
 */

#ifndef TORCH5SPRO_IP_BLOCK_H
#define TORCH5SPRO_IP_BLOCK_H 1

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/common.h"
#include "ip/crop.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which rotates a 2D blitz::array/image of a given type
        */
      template<typename T, typename U>
      void blockNoCheck(const blitz::Array<T,2>& src, U& dst, 
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

    }

    /**
      * @brief Function which perform a decomposition by block of a 2D 
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
    template<typename T, typename U>
    void block(const blitz::Array<T,2>& src, U& dst, 
      const int block_h, const int block_w, const int overlap_h, 
      const int overlap_w)
    {
      // Checks that the src array has zero base indices
      detail::assertZeroBase( src);

      // Check parameters and throw exception if required
      if( block_h<1)
        throw ParamOutOfBoundaryError("block_h", false, block_h, 1); 
      if( block_h>src.extent(1) )
        throw ParamOutOfBoundaryError("block_h", true, block_h, 
          src.extent(0)); 
      if( block_w<1)
        throw ParamOutOfBoundaryError("block_w", false, block_w, 1); 
      if( block_w>src.extent(1) )
        throw ParamOutOfBoundaryError("block_w", true, block_w, 
          src.extent(1)); 
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

      // Crop the 2D array
      detail::blockNoCheck(src, dst, block_h, block_w, overlap_h, overlap_w);
    }
  }

/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_BLOCK_H */
