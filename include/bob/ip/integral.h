/**
 * @file cxx/ip/ip/integral.h
 * @date Thu Apr 28 20:09:16 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to compute the integral image of a 2D
 *  or 3D array/image.
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

#ifndef BOB_IP_INTEGRAL_H
#define BOB_IP_INTEGRAL_H

#include "bob/core/array_assert.h"
#include "bob/core/array_index.h"
#include "bob/core/cast.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which computes the integral image of a 2D 
        *   blitz::array/image of a given type.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning No check is performed wrt. the array dimensions.
        * @param src The input blitz array
        * @param dst The output blitz array
        */
      template<typename T, typename U>
      void integralNoCheck(const blitz::Array<T,2>& src,
        blitz::Array<U,2>& dst)
      {
        dst(0,0) = bob::core::cast<U>(src(0,0));
        // Compute first row
        for(int x=1; x<src.extent(1); ++x)
          dst(0,x) = dst(0,x-1) + bob::core::cast<U>(src(0,x));
        // Compute remaining part
        for(int y=1; y<src.extent(0); ++y)
        {
          dst(y,0) = dst(y-1,0) + bob::core::cast<U>(src(y,0));
          U row_sum_cur = src(y,0);
          for(int x=1; x<src.extent(1); ++x) 
          {
            row_sum_cur += bob::core::cast<U>(src(y,x));
            dst(y,x) = dst(y-1,x) + row_sum_cur;
          }
        }
      }
    }


    /**
      * @brief Function which computes the integral image of a 2D 
      *   blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning It is the user responsability to select a suitable type
      *   for the destination array. Using a type with the same range of 
      *   value might cause out of range problems.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param addZeroBorder This requires the dst array to be 1 pixel 
      *   larger in each dimension. Besides, an extra zero pixel will be
      *   added at the beginning of each row and column
      */
    template<typename T, typename U>
    void integral(const blitz::Array<T,2>& src, blitz::Array<U,2>& dst,
      const bool addZeroBorder=false)
    {
      // Checks that the src/dst arrays have zero base indices
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      if(addZeroBorder)
      {
        blitz::TinyVector<int,2> shape = src.shape();
        shape += 1;
        bob::core::array::assertSameShape(dst,shape);
      }
      else
        bob::core::array::assertSameShape(src,dst);

      // Compute the integral image of the 2D array
      if(addZeroBorder)
      {
        for(int y=0; y<dst.extent(0); ++y)
          dst(y,0) = 0;
        for(int x=1; x<dst.extent(1); ++x)
          dst(0,x) = 0;
        blitz::Array<U,2> dst_c = 
          dst(blitz::Range(1,src.extent(0)), blitz::Range(1,src.extent(1)));
        detail::integralNoCheck(src, dst_c);
      }
      else
        detail::integralNoCheck(src, dst);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_IP_INTEGRAL_H */
