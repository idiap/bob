/**
 * @file src/cxx/ip/ip/flipflop.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to flip/flop a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_FLIPFLOP_H
#define TORCH5SPRO_IP_FLIPFLOP_H

#include "core/logging.h"
#include "core/common.h"
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
        blitz::Range  src_y( src.ubound(0), src.lbound(0), - 1 ),
                      src_x( src.lbound(1), src.ubound(1) ), 
                      dst_y( dst.lbound(0), dst.ubound(0) ),
                      dst_x( dst.lbound(1), dst.ubound(1) );
        dst(dst_y,dst_x) = src(src_y,src_x);
      }

    }


    /**
      * @brief Function which flips upside-down a 2D blitz::array/image of 
      *   a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning An exception is thrown if the dst array does not have the 
      *   same shape as the src array.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flip(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      // Check output
      Torch::core::assertSameShape(dst,src);

      // Flip the 2D array
      detail::flipNoCheck(src, dst);
    }


    /**
      * @brief Function which flips upside-down a 3D blitz::array/image of 
      *   a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @warning An exception is thrown if the dst array does not have the 
      *   same shape as the src array.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flip(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst) 
    {
      // Check output
      Torch::core::assertSameShape(dst,src);

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
      * @warning An exception is thrown if the dst array does not have the 
      *   same shape as the src array.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flop(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      // Check output
      Torch::core::assertSameShape(dst,src);

      // Flip the 2D array
      const blitz::Array<T,2> src_t = const_cast<blitz::Array<T,2>&>(src).transpose(1,0);
      blitz::Array<T,2> dst_t = dst.transpose(1,0);
      detail::flipNoCheck(src_t, dst_t);
    }


    /**
      * @brief Function which flops left-right a 3D blitz::array/image of 
      *   a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @warning An exception is thrown if the dst array does not have the 
      *   same shape as the src array.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T>
    void flop(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst) 
    {
      // Check output
      Torch::core::assertSameShape(dst,src);

      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        const blitz::Array<T,2> src_t = const_cast<blitz::Array<T,2>&>(src_slice).transpose(1,0);
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

#endif /* TORCH5SPRO_IP_FLIPFLOP_H */
