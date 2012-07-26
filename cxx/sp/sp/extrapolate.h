/**
 * @file cxx/sp/sp/extrapolate.h
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements various extrapolation techniques for 1D and 2D blitz
 * arrays.
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

#ifndef BOB_SP_EXTRAPOLATE_H
#define BOB_SP_EXTRAPOLATE_H

#include <blitz/array.h>
#include "sp/Exception.h"
#include "core/array_assert.h"

namespace bob {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    namespace Extrapolation {
      typedef enum BorderType_ {
        Zero,
        NearestNeighbour,
        Circular,
        Mirror
      } BorderType;
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
        throw ExtrapolationDstTooSmall();

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
        throw ExtrapolationDstTooSmall();

      // Sets value everywhere 
      // TODO: avoid this using the 9 blocks division as described below
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
      *   This code is longer than the simple algorithm which uses 
      *   a single loop to set the values of dst, but much 
      *   faster when working with large arrays.
      */
    template<typename T>
    void extrapolateNearest(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0))
        throw ExtrapolationDstTooSmall();

      // Computes offsets
      int offset = (dst.extent(0) - src.extent(0)) / 2;

      // Left part
      if(offset>0) 
        dst(blitz::Range(0,offset-1)) = src(0);
      // Middle part
      dst(blitz::Range(offset, offset+src.extent(0)-1)) = src;
      // Right part
      if(offset+src.extent(0)<dst.extent(0)) 
        dst(blitz::Range(offset+src.extent(0),dst.extent(0)-1)) = src(src.extent(0)-1);
    }

    /**
      * @brief Extrapolates a 2D array, using nearest neighbour
      *   This code is longer the simple algorithm which uses 
      *   two imbricated loops to set the values of dst, but much 
      *   faster when working with large arrays.
      */
    template<typename T>
    void extrapolateNearest(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0) || src.extent(1) > dst.extent(1))
        throw ExtrapolationDstTooSmall();

      // Computes offsets
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;
      
      // Defines some useful ranges
      blitz::Range r_all = blitz::Range::all();
      blitz::Range ry_middle(offset_y,offset_y+src.extent(0)-1);
      blitz::Range rx_left;
      if(offset_x>0) rx_left = blitz::Range(0,offset_x-1);
      blitz::Range rx_middle(offset_x,offset_x+src.extent(1)-1);
      blitz::Range rx_right;
      if(offset_x>0) rx_right = blitz::Range(offset_x+src.extent(1),dst.extent(1)-1);

      // Performs the extrapolation considering the 9 blocks around 
      // the middle region which is set equal to src
      // Upper part
      if(offset_y>0)
      {
        blitz::Range ry_up(0,offset_y-1);
        // Left part
        if(offset_x>0)
          dst(ry_up,rx_left) = src(0,0);
        // Middle part
        for(int j=0; j<offset_y; ++j)
          dst(j,rx_middle) = src(0,r_all);
        // Right part
        if(offset_x+src.extent(1)<dst.extent(1))
          dst(ry_up,rx_right) = src(0,src.extent(1)-1);
      }

      // Middle part
      //  Left part
      for(int i=0; i<offset_x; ++i)
        dst(ry_middle,i) = src(r_all,0);
      //  Middle part
      dst(ry_middle,rx_middle) = src;
      //  Right part
      for(int i=offset_x+src.extent(1); i<dst.extent(1); ++i)
        dst(ry_middle,i) = src(r_all,src.extent(1)-1);

      // Lower part
      if(offset_y+src.extent(0)<dst.extent(0))
      {
        blitz::Range ry_low(offset_y+src.extent(0),dst.extent(0)-1);
        // Left part
        if(offset_x>0)
          dst(ry_low,rx_left) = src(src.extent(0)-1,0);
        // Middle part
        for(int j=offset_y+src.extent(0); j<dst.extent(0); ++j)
          dst(j,rx_middle) = src(src.extent(0)-1,r_all);
        // Right part
        if(offset_x+src.extent(1)<dst.extent(1))
          dst(ry_low,rx_right) = src(src.extent(0)-1,src.extent(1)-1);
      }
    }
    

    namespace detail {
      template<typename T>
      void extrapolateCircularRec(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
      {
        // Computes offset
        int offset = (dst.extent(0) - src.extent(0)) / 2;
        int offset_left = 0;
        int offset_right = dst.extent(0)-1;
        if(offset >= src.extent(0))
        {
          offset_left = offset-src.extent(0);
          offset_right = offset+src.extent(0)+(offset-offset_left) - 1;
        }

        // Left part
        if(offset_left!=offset)
          dst(blitz::Range(offset_left,offset-1)) = 
            src(blitz::Range(src.extent(0)-(offset-offset_left),src.extent(0)-1));
        // Right part
        if(offset+src.extent(0)<=offset_right)
          dst(blitz::Range(offset+src.extent(0), offset_right)) = 
            src(blitz::Range(0,offset_right-(offset+src.extent(0))));

        // Calls this recursively if required
        if(offset_left!=0 || offset_right!=dst.extent(0)-1)
        {
          blitz::Array<T,1> dst_s = dst(blitz::Range(offset_left,offset_right));
          extrapolateCircularRec(dst_s, dst);
        }
      }

      template<typename T>
      void extrapolateCircularRec(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
      {
        // Computes offset
        int offset_y = (dst.extent(0) - src.extent(0)) / 2;
        int offset_x = (dst.extent(1) - src.extent(1)) / 2;
        int offset_y_up = 0;
        int offset_y_low = dst.extent(0)-1;
        int offset_x_left = 0;
        int offset_x_right = dst.extent(1)-1;
        if(offset_y >= src.extent(0))
        {
          offset_y_up = offset_y-src.extent(0);
          offset_y_low = offset_y+src.extent(0)+(offset_y-offset_y_up) - 1;
        }
        if(offset_x >= src.extent(1))
        {
          offset_x_left = offset_x-src.extent(1);
          offset_x_right = offset_x+src.extent(1)+(offset_x-offset_x_left) - 1;
        }

        // Defines some useful ranges
        blitz::Range r_all = blitz::Range::all();

        // Performs the extrapolation considering the 9 blocks around 
        // the middle region which is set equal to src
        // Upper part
        if(offset_y>0)
        {
          blitz::Range ry_up(offset_y_up,offset_y-1);
          blitz::Range ry_src_down(src.extent(0)-(offset_y-offset_y_up),src.extent(0)-1);
          // Left part
          if(offset_x>0)
            dst(ry_up, blitz::Range(offset_x_left,offset_x-1)) = 
              src(ry_src_down, blitz::Range(src.extent(1)-(offset_x-offset_x_left),src.extent(1)-1));
          // Middle part
          for(int j=offset_y_up; j<offset_y; ++j)
            dst(j,blitz::Range(offset_x,offset_x+src.extent(1)-1)) = src(src.extent(0)-offset_y+j,r_all);
          // Right part
          if(offset_x+src.extent(1)<dst.extent(1))
            dst(ry_up, blitz::Range(offset_x+src.extent(1),offset_x_right)) =
              src(ry_src_down, blitz::Range(0,offset_x_right-(offset_x+src.extent(1))));
        }

        // Middle part
        blitz::Range ry_mid(offset_y,offset_y+src.extent(0)-1);
        //  Left part
        if(offset_x>0)
          dst(ry_mid, blitz::Range(offset_x_left,offset_x-1)) = 
            src(r_all, blitz::Range(src.extent(1)-(offset_x-offset_x_left),src.extent(1)-1));
        //  Right part
        if(offset_x+src.extent(1)<dst.extent(1))
          dst(ry_mid, blitz::Range(offset_x+src.extent(1),offset_x_right)) =
            src(r_all, blitz::Range(0,offset_x_right-(offset_x+src.extent(1))));

        // Lower part
        if(offset_y+src.extent(0)<dst.extent(0))
        {
          blitz::Range ry_low(offset_y+src.extent(0),offset_y_low);
          blitz::Range ry_src_up(0,offset_y_low-offset_y-src.extent(0));
          // Left part
          if(offset_x>0)
            dst(ry_low, blitz::Range(offset_x_left,offset_x-1)) = 
              src(ry_src_up, blitz::Range(src.extent(1)-(offset_x-offset_x_left),src.extent(1)-1));
          // Middle part
          for(int j=offset_y+src.extent(0); j<=offset_y_low; ++j)
            dst(j,blitz::Range(offset_x,offset_x+src.extent(1)-1)) = src(j-(offset_y+src.extent(0)),r_all);
          // Right part
          if(offset_x+src.extent(1)<dst.extent(1))
            dst(ry_low, blitz::Range(offset_x+src.extent(1),offset_x_right)) =
              src(ry_src_up, blitz::Range(0,offset_x_right-(offset_x+src.extent(1))));
        }

        // Calls this recursively if required
        if(offset_y_up!=0 || offset_y_low!=dst.extent(0)-1 ||
          offset_x_left!=0 || offset_x_right!=dst.extent(1)-1)
        {
          blitz::Array<T,2> dst_s = dst(blitz::Range(offset_y_up,offset_y_low),blitz::Range(offset_x_left,offset_x_right));
          extrapolateCircularRec(dst_s, dst);
        }
      }


      template<typename T>
      void extrapolateMirrorRec(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
      {
        // Computes offset
        int offset = (dst.extent(0) - src.extent(0)) / 2;
        int offset_left = 0;
        int offset_right = dst.extent(0)-1;
        if(offset >= src.extent(0))
        {
          offset_left = offset-src.extent(0);
          offset_right = offset+src.extent(0)+(offset-offset_left) - 1;
        }

        // Left part
        if(offset_left!=offset) 
          dst(blitz::Range(offset_left,offset-1)) = 
            src(blitz::Range((offset-1-offset_left),0,-1));
        // Right part
        if(offset+src.extent(0)<=offset_right)
          dst(blitz::Range(offset+src.extent(0), offset_right)) = 
            src(blitz::Range(src.extent(0)-1,2*src.extent(0)-offset_right+offset-1,-1));

        // Call this recursively if required
        if(offset_left!=0 || offset_right!=dst.extent(0)-1)
        {
          blitz::Array<T,1> dst_s = dst(blitz::Range(offset_left,offset_right));
          extrapolateMirrorRec(dst_s, dst);
        }
      }

      template<typename T>
      void extrapolateMirrorRec(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
      {
        // Computes offset
        int offset_y = (dst.extent(0) - src.extent(0)) / 2;
        int offset_x = (dst.extent(1) - src.extent(1)) / 2;
        int offset_y_up = 0;
        int offset_y_low = dst.extent(0)-1;
        int offset_x_left = 0;
        int offset_x_right = dst.extent(1)-1;
        if(offset_y >= src.extent(0))
        {
          offset_y_up = offset_y-src.extent(0);
          offset_y_low = offset_y+src.extent(0)+(offset_y-offset_y_up) - 1;
        }
        if(offset_x >= src.extent(1))
        {
          offset_x_left = offset_x-src.extent(1);
          offset_x_right = offset_x+src.extent(1)+(offset_x-offset_x_left) - 1;
        }

        // Defines some useful ranges
        blitz::Range r_all = blitz::Range::all();

        // Performs the extrapolation considering the 9 blocks around 
        // the middle region which is set equal to src
        // Upper part
        if(offset_y>0)
        {
          blitz::Range ry_up(offset_y_up,offset_y-1);
          blitz::Range ry_src_upr(offset_y-1-offset_y_up,0,-1);
          // Left part
          if(offset_x>0)
            dst(ry_up, blitz::Range(offset_x_left,offset_x-1)) = 
              src(ry_src_upr, blitz::Range(offset_x-1-offset_x_left,0,-1));
          // Middle part
          for(int j=offset_y_up; j<offset_y; ++j)
            dst(j,blitz::Range(offset_x,offset_x+src.extent(1)-1)) = src(offset_y-1-j,r_all);
          // Right part
          if(offset_x+src.extent(1)<dst.extent(1))
            dst(ry_up, blitz::Range(offset_x+src.extent(1),offset_x_right)) =
              src(ry_src_upr, blitz::Range(src.extent(1)-1,2*src.extent(1)+offset_x-offset_x_right-1,-1));
        }

        // Middle part
        blitz::Range ry_mid(offset_y,offset_y+src.extent(0)-1);
        //  Left part
        if(offset_x>0)
          dst(ry_mid, blitz::Range(offset_x_left,offset_x-1)) = 
            src(r_all, blitz::Range(offset_x-1-offset_x_left,0,-1));
        //  Right part
        if(offset_x+src.extent(1)<dst.extent(1))
          dst(ry_mid, blitz::Range(offset_x+src.extent(1),offset_x_right)) =
            src(r_all, blitz::Range(src.extent(1)-1,2*src.extent(1)+offset_x-offset_x_right-1,-1));

        // Lower part
        if(offset_y+src.extent(0)<dst.extent(0))
        {
          blitz::Range ry_low(offset_y+src.extent(0),offset_y_low);
          blitz::Range ry_src_lowr(src.extent(0)-1,2*src.extent(0)-offset_y_low+offset_y-1,-1);
          // Left part
          if(offset_x>0)
            dst(ry_low, blitz::Range(offset_x_left,offset_x-1)) = 
              src(ry_src_lowr, blitz::Range(offset_x-1-offset_x_left,0,-1));
          // Middle part
          for(int j=offset_y+src.extent(0); j<=offset_y_low; ++j)
            dst(j,blitz::Range(offset_x,offset_x+src.extent(1)-1)) = src(2*src.extent(0)-1-(j-offset_y),r_all);
          // Right part
          if(offset_x+src.extent(1)<dst.extent(1))
            dst(ry_low, blitz::Range(offset_x+src.extent(1),offset_x_right)) =
              src(ry_src_lowr, blitz::Range(src.extent(1)-1,2*src.extent(1)+offset_x-offset_x_right-1,-1));
        }

        // Calls this recursively if required
        if(offset_y_up!=0 || offset_y_low!=dst.extent(0)-1 ||
          offset_x_left!=0 || offset_x_right!=dst.extent(1)-1)
        {
          blitz::Array<T,2> dst_s = dst(blitz::Range(offset_y_up,offset_y_low),blitz::Range(offset_x_left,offset_x_right));
          extrapolateMirrorRec(dst_s, dst);
        }
      }
    }

    /**
      * @brief Extrapolates a 1D array, using circular extrapolation
      *   This code is longer than the simple algorithm which uses 
      *   a single loop to set the values of dst, but much 
      *   faster when working with large arrays.
      */
    template<typename T>
    void extrapolateCircular(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0))
        throw ExtrapolationDstTooSmall();

      // Computes offset
      int offset = (dst.extent(0) - src.extent(0)) / 2;
      // Sets (middle) values
      dst(blitz::Range(offset,offset+src.extent(0)-1)) = src;
      // Calls the recursive function for the borders
      detail::extrapolateCircularRec(src, dst);
    }

    /**
      * @brief Extrapolates a 2D array, using circular extrapolation
      *   This code is longer than the simple algorithm which uses 
      *   a single loop to set the values of dst, but much 
      *   faster when working with large arrays.
      */
    template<typename T>
    void extrapolateCircular(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0) || src.extent(1) > dst.extent(1))
        throw ExtrapolationDstTooSmall();

      // Computes offset
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;
      // Sets (middle) values
      dst(blitz::Range(offset_y,offset_y+src.extent(0)-1), blitz::Range(offset_x,offset_x+src.extent(1)-1)) = src;
      // Calls the recursive function for the borders
      detail::extrapolateCircularRec(src, dst);
    }


    /**
      * @brief Extrapolates a 1D array, using mirroring
      *   This code is longer than the simple algorithm which uses 
      *   a single loop to set the values of dst, but much 
      *   faster when working with large arrays.
      */
    template<typename T>
    void extrapolateMirror(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0))
        throw ExtrapolationDstTooSmall();

      // Computes offset
      int offset = (dst.extent(0) - src.extent(0)) / 2;
      // Sets (middle) values
      dst(blitz::Range(offset,offset+src.extent(0)-1)) = src;
      // Calls the recursive function for the borders
      detail::extrapolateMirrorRec(src, dst);
    }

    /**
      * @brief Extrapolates a 2D array, using mirroring
      *   This code is longer than the simple algorithm which uses 
      *   a single loop to set the values of dst, but much 
      *   faster when working with large arrays.
      */
    template<typename T>
    void extrapolateMirror(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0) || src.extent(1) > dst.extent(1))
        throw ExtrapolationDstTooSmall();

      // Computes offsets
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;

      // Sets (middle) values
      dst(blitz::Range(offset_y,offset_y+src.extent(0)-1), blitz::Range(offset_x,offset_x+src.extent(1)-1)) = src;
      // Calls the recursive function for the borders
      detail::extrapolateMirrorRec(src, dst);
    }
 
  }
/**
 * @}
 */
}

#endif /* BOB_SP_INTERPOLATE_H */
