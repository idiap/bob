/**
 * @file src/cxx/ip/ip/flipflop.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to crop a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_FLIPFLOP_H
#define TORCH5SPRO_IP_FLIPFLOP_H 1

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/common.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which flips upside-down a 2D blitz::array/image of 
        *   a given type.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis). 
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        */
      template<typename T>
      void flipNoCheck(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
      {
        for( int y=0; y<src.extent(0); ++y)
          for( int x=0; x<src.extent(1); ++x)
            dst(y,x) = src( src.ubound(0)-y, x+src.lbound(0));
      }

    }


    /**
      * @brief Function which flips upside-down a 2D blitz::array/image of 
      *   a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flip(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
        dst.resize( src.extent(0), src.extent(1) );

      // Flip the 2D array
      detail::flipNoCheck(src, dst);
    }


    /**
      * @brief Function which flips upside-down a 3D blitz::array/image of 
      *   a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero 
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flip(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst) 
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 || dst.base(2) != 0 ) {
        const blitz::TinyVector<int,3> zero_base = 0;
        dst.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) || 
          dst.extent(2) != src.extent(2) )
        dst.resize( src.extent(0), src.extent(1), src.extent(2) );

      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        // Flip the 2D array
        detail::flipNoCheck(src_slice, dst_slice);
      }
    }


    /**
      * @brief Function which flops left-right a 2D blitz::array/image of 
      *   a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flop(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
        dst.resize( src.extent(0), src.extent(1) );

      // Flip the 2D array
      const blitz::Array<T,2> src_t = (src.copy()).transpose(1,0);
      blitz::Array<T,2> dst_t = dst.transpose(1,0);
      detail::flipNoCheck(src_t, dst_t);
    }


    /**
      * @brief Function which flops left-right a 3D blitz::array/image of 
      *   a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero 
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flop(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst) 
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 || dst.base(2) != 0 ) {
        const blitz::TinyVector<int,3> zero_base = 0;
        dst.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) || 
          dst.extent(2) != src.extent(2) )
        dst.resize( src.extent(0), src.extent(1), src.extent(2) );

      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        const blitz::Array<T,2> src_t = (src_slice.copy()).transpose(1,0);
        blitz::Array<T,2> dst_t = dst_slice.transpose(1,0);
        // Flip the 2D array
        detail::flipNoCheck(src_t, dst_t);
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_CROP_H */

