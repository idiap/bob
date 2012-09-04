/**
 * @file cxx/ip/ip/shear.h
 * @date Wed Mar 9 19:09:08 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to shear/skew a 2D or 3D array/image.
 * The algorithm is strongly inspired by the following article:
 * 'A Fast Algorithm for General Raster Rotation', Alan Paeth, in the
 * proceedings of Graphics Interface '86, p. 77-81.
 * The notes of Tobin Fricke about this article might also be of interest.
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

#ifndef BOB_IP_SHEAR_H
#define BOB_IP_SHEAR_H

#include <blitz/array.h>
#include "bob/core/array_assert.h"
#include "bob/core/cast.h"
#include "bob/ip/Exception.h"
#include "bob/ip/common.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which shears a 2D blitz::array/image of a given type
        *   along the X-axis.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param src_mask The input blitz array mask, specifying the valid
        *   pixels of src.
        * @param dst The output blitz array
        * @param dst_mask The output blitz array mask, specifying the valid
        *   pixels of dst.
        * @param shear The shear parameter in the matrix [1 shear; 0 1]
        * @param antialias Whether antialiasing is used or not
        */
      template<typename T, bool mask>
      void shearXNoCheck(
        const blitz::Array<T,2>& src, const blitz::Array<bool,2>& src_mask,
        blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask, 
        const double shear, const bool antialias)
      {
        // Compute center coordinates in src and dst image
        double y_c = (src.extent(0) - 1)/ 2.;
        double x_c_src = (src.extent(1) - 1)/ 2.;
        double x_c_dst = (dst.extent(1) - 1)/ 2.;

        // If shear is equal to zero, we just need to do a simple copy
        if(shear == 0.) {
          for( int y=0; y<src.extent(0); ++y) 
            for( int x=0; x<src.extent(1); ++x) 
              dst(y,x) = bob::core::cast<double>(src(y,x));
          if(mask)
            detail::copyNoCheck(src_mask,dst_mask);
          return;
        }

        // Initialize dst to background value
        dst = 0;
        dst_mask = false;

        // Loop over the rows and skew them horizontally
        for( int y=0; y<src.extent(0); ++y) {
          // Determine the skew offset wrt. the center of the input
          double skew = shear * (y - y_c);
          // Determine the direction and make the skew positive
          bool dir_right;
          if( skew > 0.)
            dir_right = true;
          else {
            dir_right = false;
            skew = -skew;
          }
          // Compute the floor of the skew
          int skew_i;
          if( antialias)
            skew_i = floor(skew);
          else
            skew_i = floor(skew+0.5);
          // Compute the residual of the skew
          double skew_f = skew - skew_i;
          double old_residual = 0.;

          // Transfer pixels right-to-left
          if( dir_right) {
            // Loop over all the input pixels of the row
            for( int x=src.extent(1)-1; x>=0; --x) {
              double pixel = static_cast<double>(src(y,x));
              double residual;
              if( antialias )
                residual = pixel * skew_f;
              else 
                residual = 0.;
              pixel = (pixel - residual) + old_residual;
              // Determine x-location on dst row
              int x_dst = ceil(x - x_c_src + x_c_dst - skew_i-0.5);
              if( x_dst >= 0 && x_dst < dst.extent(1) ) {
                dst(y,x_dst) = pixel;
                if(mask) 
                  dst_mask(y,x_dst) = ( x==src.extent(1)-1 ?
                      (antialias && skew_f !=0. ? false : src_mask(y,x))
                    : src_mask(y,x) && src_mask(y,x+1));
              }
              old_residual = residual;
            }
            // Add remaining residual if possible
            double next_ind = -x_c_src + x_c_dst - skew_i - 1 - 0.5;
            if( ceil(next_ind) >= 0) {
              dst(y,(int)ceil(next_ind)) = old_residual;
              if(mask) 
                dst_mask(y,(int)ceil(next_ind)) = false;
            }
          }
          // Transfer pixels left-to-right
          else {
            // Loop over all the input pixels of the row
            for( int x=0; x<src.extent(1); ++x) {
              double pixel = static_cast<double>(src(y,x));
              double residual;
              if( antialias )
                residual = pixel * skew_f;
              else 
                residual = 0.;
              pixel = (pixel - residual) + old_residual;
              int x_dst = ceil(x - x_c_src + x_c_dst + skew_i-0.5);
              if( x_dst >= 0 && x_dst < dst.extent(1) ) {
                dst(y,x_dst) = pixel;
                if(mask) 
                  dst_mask(y,x_dst) = (x==0 ? 
                      (antialias && skew_f !=0. ? false : src_mask(y,x))
                    : src_mask(y,x) && src_mask(y,x-1));
              }
              old_residual = residual;
            }
            // Add remaining residual if possible
            double next_ind = 
              -x_c_src + x_c_dst + skew_i + src.extent(1) - 0.5;
            if( ceil(next_ind) < dst.extent(1)) {
              dst(y,(int)ceil(next_ind)) = old_residual;
              if(mask) 
                dst_mask(y,(int)ceil(next_ind)) = false;
            }
          } 
        }    
      }

    }


    /**
      * @brief Return the shape of the output image/array, when performing
      * a shearing along the X-axis, with the given input image/array and 
      * shear parameter.
      * @param src the input array
      * @param shear The shear parameter in the matrix [1 shear; 0 1]
      */
    template <typename T>
    const blitz::TinyVector<int,2> getShearXShape( 
      const blitz::Array<T,2>& src, const double shear)
    {
      // Compute the required output size when applying shearX
      blitz::TinyVector<int,2> res;
      res(0) = src.extent(0);
      res(1) = src.extent(1) + floor(fabs(shear)*(src.extent(0)-1)+0.5);
      return res;
    }

    /**
      * @brief Return the shape of the output image/array, when performing
      * a shearing along the Y-axis, with the given input image/array and 
      * shear parameter.
      * @param src the input array
      * @param shear The shear parameter in the matrix [1 shear; 0 1]
      */
    template <typename T>
    const blitz::TinyVector<int,2> getShearYShape( 
      const blitz::Array<T,2>& src, const double shear)
    {
      // Compute the required output size when applying shearX
      blitz::TinyVector<int,2> res;
      res(0) = src.extent(0) + floor(fabs(shear)*(src.extent(1)-1)+0.5);
      res(1) = src.extent(1);
      return res;
    }


    /**
      * @brief Function which shears a 2D blitz::array/image of a given type
      *   along the X-axis.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param shear The shear parameter in the matrix [1 shear; 0 1]
      * @param antialias Whether antialiasing should be used or not 
      */
    template<typename T>
    void shearX(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst, 
      const double shear, const bool antialias=true)
    {
      // Check input
      bob::core::array::assertZeroBase(src);
      // Check output
      bob::core::array::assertZeroBase(dst);
      const blitz::TinyVector<int,2> shape = getShearXShape(src, shear); 
      bob::core::array::assertSameShape(dst, shape);

      // Call the shearXNoCheck function
      blitz::Array<bool,2> src_mask, dst_mask;
      detail::shearXNoCheck<T,false>( src, src_mask, dst, dst_mask, shear, 
        antialias);
    }

    /**
      * @brief Function which shears a 2D blitz::array/image of a given type
      *   along the X-axis.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param src_mask The input blitz array mask, specifying the valid
      *   pixels of src.
      * @param dst The output blitz array
      * @param dst_mask The output blitz array mask, specifying the valid
      *   pixels of dst.
      * @param shear The shear parameter in the matrix [1 shear; 0 1]
      * @param antialias Whether antialiasing should be used or not 
      */
    template<typename T>
    void shearX(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask,
      blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask,
      const double shear, const bool antialias=true)
    {
      // Check input
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(src_mask);
      bob::core::array::assertSameShape(src, src_mask);
      // Check output
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertZeroBase(dst_mask);
      bob::core::array::assertSameShape(dst, dst_mask);
      const blitz::TinyVector<int,2> shape = getShearXShape(src, shear); 
      bob::core::array::assertSameShape(dst, shape);

      // Call the shearXNoCheck function
      detail::shearXNoCheck<T,true>( src, src_mask, dst, dst_mask, shear, 
        antialias);
    }

    /**
      * @brief Function which shears a 2D blitz::array/image of a given type
      *   along the Y-axis.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param shear The shear parameter in the matrix [1 0; shear 1]
      * @param antialias Whether antialiasing should be used or not 
      */
    template<typename T>
    void shearY(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst, 
      const double shear, const bool antialias=true)
    {
      // Check input
      bob::core::array::assertZeroBase(src);
      // Check output
      bob::core::array::assertZeroBase(dst);
      const blitz::TinyVector<int,2> shape = getShearYShape(src, shear); 
      bob::core::array::assertSameShape(dst, shape);

      // Create transposed view arrays for both src and dst
      const blitz::Array<T,2> src_transpose = (src.copy()).transpose(1,0);
      blitz::Array<double,2> dst_transpose = dst.transpose(1,0);

      // Call the shearXNoCheck function
      blitz::Array<bool,2> src_mask_transpose, dst_mask_transpose; 
      detail::shearXNoCheck<T,false>( src_transpose, src_mask_transpose,
        dst_transpose, dst_mask_transpose, shear, antialias);
    }

    /**
      * @brief Function which shears a 2D blitz::array/image of a given type
      *   along the Y-axis.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param src_mask The input blitz array mask, specifying the valid
      *   pixels of src.
      * @param dst The output blitz array
      * @param dst_mask The output blitz array mask, specifying the valid
      *   pixels of dst.
      * @param shear The shear parameter in the matrix [1 0; shear 1]
      * @param antialias Whether antialiasing should be used or not 
      */
    template<typename T>
    void shearY(const blitz::Array<T,2>& src, const blitz::Array<bool,2>& src_mask,
      blitz::Array<double,2>& dst, blitz::Array<bool,2>& dst_mask,
      const double shear, const bool antialias=true)
    {
      // Check input
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(src_mask);
      bob::core::array::assertSameShape(src, src_mask);
      // Check output
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertZeroBase(dst_mask);
      bob::core::array::assertSameShape(dst, dst_mask);
      const blitz::TinyVector<int,2> shape = getShearYShape(src, shear); 
      bob::core::array::assertSameShape(dst, shape);

      // Create transposed view arrays for both src and dst
      const blitz::Array<T,2> src_transpose = (src.copy()).transpose(1,0);
      const blitz::Array<bool,2> src_mask_transpose = 
        (src_mask.copy()).transpose(1,0);
      blitz::Array<double,2> dst_transpose = dst.transpose(1,0);
      blitz::Array<bool,2> dst_mask_transpose = dst_mask.transpose(1,0);

      // Call the shearXNoCheck function
      detail::shearXNoCheck<T,true>( src_transpose, src_mask_transpose, 
        dst_transpose, dst_mask_transpose, shear, antialias);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_IP_SHEAR_H */
