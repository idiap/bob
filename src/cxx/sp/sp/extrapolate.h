/**
 * @file cxx/sp/sp/extrapolate.h
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements various extrapolation techniques for 1D and 2D blitz
 * arrays.
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_SP_EXTRAPOLATE_H
#define BOB_SP_EXTRAPOLATE_H

#include <blitz/array.h>
#include "core/Exception.h"
#include "core/array_assert.h"
#include "core/array_index.h"

namespace bob {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    namespace Extrapolation {
      enum BorderType {
        Zero,
        NearestNeighbour,
        Circular,
        Mirror
      }; 
    }

    /**
      * @brief Extrapolates a 1D array, padding with a constant
      */
    template<typename T>
    void extrapolateConstant(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst, 
      const T value)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0))
        throw bob::core::Exception();

      // Sets value everywhere
      dst = value;
      // Computes offset and range
      int offset = (dst.extent(0) - src.extent(0)) / 2;
      blitz::Range dst_range(offset, offset+src.extent(0)-1);
      blitz::Array<T,1> dst_slice = dst(dst_range);
      // Copies data from src array
      dst_slice = src;
    }

    /**
      * @brief Extrapolates a 2D array, padding with a constant
      */
    template<typename T>
    void extrapolateConstant(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const T value)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0) || src.extent(1) > dst.extent(1))
        throw bob::core::Exception();

      // Sets value everywhere
      dst = value;
      // Computes offsets and ranges
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;
      blitz::Range dst_range_y(offset_y, offset_y+src.extent(0)-1);
      blitz::Range dst_range_x(offset_x, offset_x+src.extent(1)-1);
      blitz::Array<T,2> dst_slice = dst(dst_range_y,dst_range_x);
      // Copies data from src array
      dst_slice = src;
    }


    /**
      * @brief Extrapolates a 1D array, using zero padding
      */
    template<typename T>
    void extrapolateZero(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Calls extrapolate with the constant set to 0
      T zero = 0;
      bob::sp::extrapolateConstant(src, dst, zero);
    }

    /**
      * @brief Extrapolates a 2D array, using zero padding
      */
    template<typename T>
    void extrapolateZero(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Calls extrapolate with the constant set to 0
      T zero = 0;
      bob::sp::extrapolateConstant(src, dst, zero);
    }


    /**
      * @brief Extrapolates a 1D array, using nearest neighbour
      */
    template<typename T>
    void extrapolateNearest(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset = (dst.extent(0) - src.extent(0)) / 2;
      for(int i=0; i<dst.extent(0); ++i)
        dst(i) = src( bob::core::array::keepInRange(i-offset, 0, src.extent(0)-1));
    }

    /**
      * @brief Extrapolates a 2D array, using nearest neighbour
      */
    template<typename T>
    void extrapolateNearest(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;
      for(int j=0; j<dst.extent(0); ++j)
        for(int i=0; i<dst.extent(1); ++i)
          dst(j,i) = src( bob::core::array::keepInRange(j-offset_y, 0, src.extent(0)-1),
                          bob::core::array::keepInRange(i-offset_x, 0, src.extent(1)-1));
    }
    

    /**
      * @brief Extrapolates a 1D array, using circular extrapolation
      */
    template<typename T>
    void extrapolateCircular(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      // Computes offset
      int offset = (dst.extent(0) - src.extent(0)) / 2;
      // Sets left values
      int s = src.extent(0);
      for(int i=0; i<dst.extent(0); ++i)
        dst(i) = src( ((i-offset % s) + s) % s );
    }

    /**
      * @brief Extrapolates a 2D array, using circular extrapolation
      */
    template<typename T>
    void extrapolateCircular(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      // Computes offset
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;
      // Sets values
      int s_y = src.extent(0);
      int s_x = src.extent(1);
      for(int j=0; j<dst.extent(0); ++j)
        for(int i=0; i<dst.extent(1); ++i)
          dst(j,i) = src( ((j-offset_y % s_y) + s_y) % s_y, ((i-offset_x % s_x) + s_x) % s_x );
    }


    /**
      * @brief Extrapolates a 1D array, using mirroring
      */
    template<typename T>
    void extrapolateMirror(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset = (dst.extent(0) - src.extent(0)) / 2;

      // Sets left values
      for(int i=0; i<dst.extent(0); ++i)
        dst(i) = src( bob::core::array::mirrorInRange(i-offset, 0, src.extent(0)-1));
    }

    /**
      * @brief Extrapolates a 2D array, using mirroring
      */
    template<typename T>
    void extrapolateMirror(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;

      // Sets left values
      for(int j=0; j<dst.extent(0); ++j)
        for(int i=0; i<dst.extent(1); ++i)
          dst(j,i) = src( bob::core::array::mirrorInRange(j-offset_y, 0, src.extent(0)-1),
                          bob::core::array::mirrorInRange(i-offset_x, 0, src.extent(1)-1));
    }
 
  }
/**
 * @}
 */
}

#endif /* BOB_SP_INTERPOLATE_H */
