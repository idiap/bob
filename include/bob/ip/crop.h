/**
 * @file bob/ip/crop.h
 * @date Mon Mar 7 18:00:00 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to crop a 2D or 3D array/image.
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

#ifndef BOB_IP_CROP_H
#define BOB_IP_CROP_H

#include "bob/core/array_assert.h"
#include "bob/core/array_index.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Checks the given cropping parameters wrt. given input 
        *   dimensions, and throws an exception if one part of the cropping
        *   area is outside the boundary of the source array.
        * @param crop_y The y-offset of the top left corner of the cropping 
        *   area wrt. to the y-index of the top left corner of the 
        *   blitz::array.
        * @param crop_x The x-offset of the top left corner of the cropping 
        *   area wrt. to the x-index of the top left corner of the 
        *   blitz::array.
        * @param crop_h The desired height of the cropped blitz::array.
        * @param crop_w The desired width of the cropped blitz::array.
        * @param src_height The height of the input image
        * @param src_width The width of the input image
        */
      void cropParameterCheck( const int crop_y, const int crop_x,
        const size_t crop_h, const size_t crop_w, const size_t src_height, 
        const size_t src_width);

      /**
        * @brief Function which crops a 2D blitz::array/image of a given type,
        *   and references to the data of the src array.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param crop_y The y-offset of the top left corner of the cropping 
        *   area wrt. to the y-index of the top left corner of the 
        *   blitz::array.
        * @param crop_x The x-offset of the top left corner of the cropping 
        *   area wrt. to the x-index of the top left corner of the 
        *   blitz::array.
        * @param crop_h The desired height of the cropped blitz::array.
        * @param crop_w The desired width of the cropped blitz::array.
        */
      template<typename T>
      void cropNoCheckReference(const blitz::Array<T,2>& src, 
        blitz::Array<T,2>& dst, const int crop_y, const int crop_x, 
        const size_t crop_h, const size_t crop_w)
      {
        blitz::Range ry( crop_y, crop_y+crop_h-1);
        blitz::Range rx( crop_x, crop_x+crop_w-1);
        dst.reference( src( ry, rx));
      }

      /**
        * @brief Function which crops a 2D blitz::array/image of a given type.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @param src The input blitz array
        * @param src_mask The input blitz array mask, specifying the valid
        *   pixels of src.
        * @param dst The output blitz array
        * @param dst_mask The input blitz array mask, specifying the valid
        *   pixels of dst.
        * @param crop_y The y-offset of the top left corner of the cropping 
        *   area wrt. to the y-index of the top left corner of the 
        *   blitz::array.
        * @param crop_x The x-offset of the top left corner of the cropping 
        *   area wrt. to the x-index of the top left corner of the 
        *   blitz::array.
        * @param crop_h The desired height of the cropped blitz::array.
        * @param crop_w The desired width of the cropped blitz::array.
        * @param zero_out Whether the cropping area which is out of the 
        *   boundary of the input blitz array should be filled with zero 
        *   values, or with the intensity of the closest pixel in the 
        *   neighbourhood.
        */
      template<typename T, bool mask>
      void cropNoCheck(const blitz::Array<T,2>& src, 
        const blitz::Array<bool,2>& src_mask, blitz::Array<T,2>& dst, 
        blitz::Array<bool,2>& dst_mask,
        const int crop_y, const int crop_x, const size_t crop_h, 
        const size_t crop_w, const bool zero_out)
      {
        bool is_y_out;
        int y_src, x_src;
        for( int y=0; y<(int)crop_h; ++y) {
          is_y_out = y+crop_y<0 || y+crop_y>=src.extent(0);
          y_src = bob::core::array::keepInRange( y+crop_y, 0, 
                                                 src.extent(0)-1);
          for( int x=0; x<(int)crop_w; ++x) {
            if( is_y_out || x+crop_x<0 || x+crop_x>=src.extent(1) )
            {
              x_src = bob::core::array::keepInRange( x+crop_x, 0, 
                                                     src.extent(1)-1);
              dst(y,x) = (zero_out ? 0 : src( y_src, x_src) );
              if( mask )
                dst_mask(y,x) = false;
            }
            else {
              dst(y,x) = src( y+crop_y, x+crop_x);
              if( mask )
                dst_mask(y,x) = src_mask( y+crop_y, x+crop_x);
              
            } 
          }
        }
      }

    }


    /**
      * @brief Function which crops a 2D blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param crop_y The y-offset of the top left corner of the cropping 
      *   area wrt. to the y-index of the top left corner of the 
      *   blitz::array.
      * @param crop_x The x-offset of the top left corner of the cropping 
      *   area wrt. to the x-index of the top left corner of the 
      *   blitz::array.
      * @param crop_h The desired height of the cropped blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      */
    template<typename T>
    void cropReference(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int crop_y, const int crop_x, const size_t crop_h, 
      const size_t crop_w)
    {
      // Check parameters and throw exception if required
      detail::cropParameterCheck( crop_y, crop_x, crop_h, crop_w, 
        src.extent(0), src.extent(1) );
      // Checks that the src array has zero base indices
      bob::core::array::assertZeroBase( src);

      // Crop the 2D array
      detail::cropNoCheckReference(src, dst, crop_y, crop_x, crop_h, crop_w);
    }

    /**
      * @brief Function which crops a 2D blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param crop_y The y-offset of the top left corner of the cropping 
      *   area wrt. to the y-index of the top left corner of the 
      *   blitz::array.
      * @param crop_x The x-offset of the top left corner of the cropping 
      *   area wrt. to the x-index of the top left corner of the 
      *   blitz::array. 
      * @param crop_h The desired height of the cropped blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      * @param allow_out Whether an exception should be raised or not if a 
      *   part of the cropping area is out of the boundary of the input 
      *   blitz array.
      * @param zero_out Whether the cropping area which is out of the 
      *   boundary of the input blitz array should be filled with zero 
      *   values, or with the intensity of the closest pixel in the 
      *   neighbourhood.
      */
    template<typename T>
    void crop(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int crop_y, const int crop_x, const size_t crop_h, 
      const size_t crop_w, const bool allow_out=false, 
      const bool zero_out=false)
    {
      // Check parameters and throw exception if required
      if(!allow_out) 
        detail::cropParameterCheck( crop_y, crop_x, crop_h, crop_w, 
          src.extent(0), src.extent(1) );
      // Check input 
      bob::core::array::assertZeroBase(src);
      // Check output
      const blitz::TinyVector<int,2> shape(crop_h,crop_w);
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertSameShape(dst, shape);
    
      // Crop the 2D array
      blitz::Array<bool,2> src_mask, dst_mask; 
      detail::cropNoCheck<T,false>(src, src_mask, dst, dst_mask, crop_y, 
        crop_x, crop_h, crop_w, zero_out);
    }


    /**
      * @brief Function which crops a 3D blitz::array/image of a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param crop_y The y-offset of the top left corner of the cropping 
      *   area wrt. to the y-index of the top left corner of the 
      *   blitz::array.
      * @param crop_x The x-offset of the top left corner of the cropping 
      *   area wrt. to the x-index of the top left corner of the 
      *   blitz::array. 
      * @param crop_h The desired height of the cropped blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      * @param allow_out Whether an exception should be raised or not if a 
      *   part of the cropping area is out of the boundary of the input 
      *   blitz array.
      * @param zero_out Whether the cropping area which is out of the 
      *   boundary of the input blitz array should be filled with zero 
      *   values, or with the intensity of the closest pixel in the 
      *   neighbourhood.
      */
    template<typename T>
    void crop(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst, 
      const int crop_y, const int crop_x, const size_t crop_h, 
      const size_t crop_w, const bool allow_out=false, 
      const bool zero_out=false)
    {
      // Check parameters and throw exception if required
      if(!allow_out) 
        detail::cropParameterCheck( crop_y, crop_x, crop_h, crop_w, 
          src.extent(1), src.extent(2) );
      // Check input
      bob::core::array::assertZeroBase( src);
      // Check output
      const blitz::TinyVector<int,3> shape(src.extent(0), (int)crop_h, 
        (int)crop_w);
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertSameShape(dst, shape);
 
      blitz::Array<bool,2> src_mask, dst_mask; 
      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        // Crop the 2D array
        detail::cropNoCheck<T,false>(src_slice, src_mask, dst_slice, dst_mask,
          crop_y, crop_x, crop_h, crop_w, zero_out);
      }
    }

    /**
      * @brief Function which crops a 2D blitz::array/image of a given type,
      *   taking into consideration masks. Masks are used to specify which
      *   pixels are 'valid' in the input and output arrays/images.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param src_mask The input blitz array mask, specifying the valid
      *   pixels of src.
      * @param dst The output blitz array
      * @param dst_mask The output blitz array mask, specifying the valid
      *   pixels of dst.
      * @param crop_x The x-offset of the top left corner of the cropping 
      *   area wrt. to the x-index of the top left corner of the 
      *   blitz::array.
      * @param crop_y The y-offset of the top left corner of the cropping
      *   area wrt. to the y-index of the top left corner of the 
      *   blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      * @param crop_h The desired height of the cropped blitz::array.
      * @param allow_out Whether an exception should be raised or not if a
      *   part of the cropping area is out of the boundary of the input 
      *   blitz array.
      * @param zero_out Whether the cropping area which is out of the 
      *   boundary of the input blitz array should be filled with zero 
      *   values, or with the intensity of the closest pixel in the 
      *   neighbourhood.
      */
    template<typename T>
    void crop(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<T,2>& dst, 
      blitz::Array<bool,2>& dst_mask, const int crop_y, const int crop_x, 
      const size_t crop_h, const size_t crop_w, const bool allow_out=false, 
      const bool zero_out=false)
    {
      // Check parameters and throw exception if required
      if(!allow_out) 
        detail::cropParameterCheck( crop_y, crop_x, crop_h, crop_w, 
          src.extent(0), src.extent(1) );
      // Check input 
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(src_mask);
      bob::core::array::assertSameShape(src, src_mask);
      // Check output
      const blitz::TinyVector<int,2> shape((int)crop_h, (int)crop_w);
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertZeroBase(dst_mask);
      bob::core::array::assertSameShape(dst, dst_mask);
      bob::core::array::assertSameShape(dst, shape);
    
      // Crop the 2D array
      detail::cropNoCheck<T,true>(src, src_mask, dst, dst_mask, crop_y, 
        crop_x, crop_h, crop_w, zero_out);
    }


    /**
      * @brief Function which crops a 3D blitz::array/image of a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @param src The input blitz array
      * @param src_mask The input blitz array mask, specifying the valid
      *   pixels of src.
      * @param dst The output blitz array
      * @param dst_mask The output blitz array mask, specifying the valid
      *   pixels of dst.
      * @param crop_y The y-offset of the top left corner of the cropping 
      *   area wrt. to the y-index of the top left corner of the 
      *   blitz::array.
      * @param crop_x The x-offset of the top left corner of the cropping 
      *   area wrt. to the x-index of the top left corner of the 
      *   blitz::array. 
      * @param crop_h The desired height of the cropped blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      * @param allow_out Whether an exception should be raised or not if a 
      *   part of the cropping area is out of the boundary of the input 
      *   blitz array.
      * @param zero_out Whether the cropping area which is out of the 
      *   boundary of the input blitz array should be filled with zero 
      *   values, or with the intensity of the closest pixel in the 
      *   neighbourhood.
      */
    template<typename T>
    void crop(const blitz::Array<T,3>& src, 
      const blitz::Array<bool,3>& src_mask, blitz::Array<T,3>& dst, 
      blitz::Array<bool,3>& dst_mask, const int crop_y, 
      const int crop_x, const size_t crop_h, const size_t crop_w,
      const bool allow_out=false, const bool zero_out=false)
    {
      // Check parameters and throw exception if required
      if(!allow_out) 
        detail::cropParameterCheck( crop_y, crop_x, crop_h, crop_w, 
          src.extent(1), src.extent(2) );
      // Check input
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(src_mask);
      bob::core::array::assertSameShape(src, src_mask);
      // Check output
      const blitz::TinyVector<int,3> shape(src.extent(0), (int)crop_h,
        (int)crop_w);
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertZeroBase(dst_mask);
      bob::core::array::assertSameShape(dst, dst_mask);
      bob::core::array::assertSameShape(dst, shape);
 
      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        const blitz::Array<bool,2> src_mask_slice = 
          src_mask( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<bool,2> dst_mask_slice = 
          dst_mask( p, blitz::Range::all(), blitz::Range::all() );
        // Crop the 2D array
        detail::cropNoCheck<T,true>(src_slice, src_mask_slice, dst_slice,
          dst_mask_slice, crop_y, crop_x, crop_h, crop_w, zero_out);
      }
    }
  }
/**
 * @}
 */
}

#endif /* BOB_IP_CROP_H */
