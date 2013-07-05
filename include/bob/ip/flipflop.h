/**
 * @file bob/ip/flipflop.h
 * @date Mon Mar 14 16:31:07 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to flip/flop a 2D or 3D array/image.
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

#ifndef BOB_IP_FLIPFLOP_H
#define BOB_IP_FLIPFLOP_H

#include "bob/core/assert.h"

namespace bob {
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
      bob::core::array::assertSameShape(dst,src);

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
      bob::core::array::assertSameShape(dst,src);

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
      bob::core::array::assertSameShape(dst,src);

      // Flip the 2D array
      const blitz::Array<T,2> src_t =
        const_cast<blitz::Array<T,2>&>(src).transpose(1,0);
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
      bob::core::array::assertSameShape(dst,src);

      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice =
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice =
          dst( p, blitz::Range::all(), blitz::Range::all() );
        const blitz::Array<T,2> src_t =
          const_cast<blitz::Array<T,2>&>(src_slice).transpose(1,0);
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

#endif /* BOB_IP_FLIPFLOP_H */
