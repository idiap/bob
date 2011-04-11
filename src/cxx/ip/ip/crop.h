/**
 * @file src/cxx/ip/ip/crop.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to crop a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_CROP_H
#define TORCH5SPRO_IP_CROP_H

#include "core/logging.h"
#include "ip/Exception.h"
#include "core/array_assert.h"
#include "core/array_index.h"

namespace tca = Torch::core::array;

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which crops a 2D blitz::array/image of a given type.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param crop_x The x-offset of the top left corner of the cropping area 
        * wrt. to the x-index of the top left corner of the blitz::array.
        * @param crop_y The y-offset of the top left corner of the cropping area 
        * wrt. to the y-index of the top left corner of the blitz::array.
        * @param crop_w The desired width of the cropped blitz::array.
        * @param crop_h The desired height of the cropped blitz::array.
        */
      template<typename T>
      void cropNoCheckReference(const blitz::Array<T,2>& src, 
        blitz::Array<T,2>& dst, const int crop_y, const int crop_x, 
        const int crop_h, const int crop_w)
      {
        blitz::Range ry( crop_y, crop_y+crop_h-1);
        blitz::Range rx( crop_x, crop_x+crop_w-1);
        dst.reference( src( ry, rx));
      }

      /**
        * @brief Function which crops a 2D blitz::array/image of a given type.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning The destination array will contain a reference to the 
        *   cropped area of the source array
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param crop_x The x-offset of the top left corner of the cropping area 
        * wrt. to the x-index of the top left corner of the blitz::array.
        * @param crop_y The y-offset of the top left corner of the cropping area 
        * wrt. to the y-index of the top left corner of the blitz::array.
        * @param crop_w The desired width of the cropped blitz::array.
        * @param crop_h The desired height of the cropped blitz::array.
        * @param zero_out Whether the cropping area which is out of the boundary
        * of the input blitz array should be filled with zero values, or with 
        * the intensity of the closest pixel in the neighbourhood.
        */
      template<typename T>
      void cropNoCheck(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
        const int crop_x, const int crop_y, const int crop_w, const int crop_h,
        const bool zero_out)
      {
        bool is_y_out;
        int y_src, x_src;
        for( int y=0; y<crop_h; ++y) {
          is_y_out = y+crop_y<0 || y+crop_y>=src.extent(0);
          y_src = tca::keepInRange( y+crop_y, 0, src.extent(0)-1);
          for( int x=0; x<crop_w; ++x) {
            if( is_y_out || x+crop_x<0 || x+crop_x>=src.extent(1) ) {
              x_src = tca::keepInRange( x+crop_x, 0, src.extent(1)-1);
              dst(y,x) = (zero_out ? 0 : 
                src( y_src+src.lbound(0), x_src+src.lbound(1)) );
            }
            else
              dst(y,x) = src( y+crop_y+src.lbound(0), x+crop_x+src.lbound(1));
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
      * @param crop_x The x-offset of the top left corner of the cropping area 
      * wrt. to the x-index of the top left corner of the blitz::array.
      * @param crop_y The y-offset of the top left corner of the cropping area 
      * wrt. to the y-index of the top left corner of the blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      * @param crop_h The desired height of the cropped blitz::array.
      */
    template<typename T>
    void cropReference(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int crop_y, const int crop_x, const int crop_h, const int crop_w)
    {
      // Checks that the src array has zero base indices
      tca::assertZeroBase( src);

      // Check parameters and throw exception if required
      if( crop_x<0 ) {
        throw ParamOutOfBoundaryError("crop_x", false, crop_x, 0);
      }
      if( crop_y<0) {
        throw ParamOutOfBoundaryError("crop_y", false, crop_y, 0);
      }
      if( crop_w<0) {
        throw ParamOutOfBoundaryError("crop_w", false, crop_w, 0);
      }
      if( crop_h<0) {
        throw ParamOutOfBoundaryError("crop_h", false, crop_h, 0);
      }
      if( crop_x+crop_w>src.extent(1)) {
        throw ParamOutOfBoundaryError("crop_x+crop_w", true, crop_x+crop_w, 
          src.extent(1) );
      }
      if( crop_y+crop_h>src.extent(0)) {
        throw ParamOutOfBoundaryError("crop_y+crop_h", true, crop_y+crop_h, 
          src.extent(0) );
      }
    
      // Crop the 2D array
      detail::cropNoCheckReference(src, dst, crop_y, crop_x, crop_h, crop_w);
    }

    /**
      * @brief Function which crops a 2D blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param crop_x The x-offset of the top left corner of the cropping area 
      * wrt. to the x-index of the top left corner of the blitz::array.
      * @param crop_y The y-offset of the top left corner of the cropping area 
      * wrt. to the y-index of the top left corner of the blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      * @param crop_h The desired height of the cropped blitz::array.
      * @param allow_out Whether an exception should be raised or not if a part
      * of the cropping area is out of the boundary of the input blitz array.
      * @param zero_out Whether the cropping area which is out of the boundary
      * of the input blitz array should be filled with zero values, or with 
      * the intensity of the closest pixel in the neighbourhood.
      */
    template<typename T>
    void crop(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int crop_x, const int crop_y, const int crop_w, const int crop_h,
      const bool allow_out=false, const bool zero_out=false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != crop_h || dst.extent(1) != crop_w )
        dst.resize( crop_h, crop_w );
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }

      // Check parameters and throw exception if required
      if(!allow_out) 
      {
        if( crop_x<0 ) {
          throw ParamOutOfBoundaryError("crop_x", false, crop_x, 0);
        }
        if( crop_y<0) {
          throw ParamOutOfBoundaryError("crop_y", false, crop_y, 0);
        }
        if( crop_w<0) {
          throw ParamOutOfBoundaryError("crop_w", false, crop_w, 0);
        }
        if( crop_h<0) {
          throw ParamOutOfBoundaryError("crop_h", false, crop_h, 0);
        }
        if( crop_x+crop_w>src.extent(1)) {
          throw ParamOutOfBoundaryError("crop_x+crop_w", true, crop_x+crop_w, 
            src.extent(1) );
        }
        if( crop_y+crop_h>src.extent(0)) {
          throw ParamOutOfBoundaryError("crop_y+crop_h", true, crop_y+crop_h, 
            src.extent(0) );
        }
      }
    
      // Crop the 2D array
      detail::cropNoCheck(src, dst, crop_x, crop_y, crop_w, crop_h, zero_out);
    }


    /**
      * @brief Function which crops a 3D blitz::array/image of a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero 
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param crop_x The x-offset of the top left corner of the cropping area 
      * wrt. to the x-index of the top left corner of the blitz::array.
      * @param crop_y The y-offset of the top left corner of the cropping area 
      * wrt. to the y-index of the top left corner of the blitz::array.
      * @param crop_w The desired width of the cropped blitz::array.
      * @param crop_h The desired height of the cropped blitz::array.
      * @param allow_out Whether an exception should be raised or not if a part
      * of the cropping area is out of the boundary of the input blitz array.
      * @param zero_out Whether the cropping area which is out of the boundary
      * of the input blitz array should be filled with zero values, or with 
      * the intensity of the closest pixel in the neighbourhood.
      */
    template<typename T>
    void crop(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst, 
      const int crop_x, const int crop_y, const int crop_w, const int crop_h,
      const bool allow_out=false, const bool zero_out=false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != crop_h || 
          dst.extent(2) != crop_w )
        dst.resize( src.extent(0), crop_h, crop_w );
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 || dst.base(2) != 0 ) {
        const blitz::TinyVector<int,3> zero_base = 0;
        dst.reindexSelf( zero_base );
      }

      // Check parameters and throw exception if required
      if(!allow_out) 
      {
        if( crop_x<0 ) {
          throw ParamOutOfBoundaryError("crop_x", false, crop_x, 0);
        }
        if( crop_y<0) {
          throw ParamOutOfBoundaryError("crop_y", false, crop_y, 0);
        }
        if( crop_w<0) {
          throw ParamOutOfBoundaryError("crop_w", false, crop_w, 0);
        }
        if( crop_h<0) {
          throw ParamOutOfBoundaryError("crop_h", false, crop_h, 0);
        }
        if( crop_x+crop_w>src.extent(2)) {
          throw ParamOutOfBoundaryError("crop_x+crop_w", true, crop_x+crop_w, 
            src.extent(2) );
        }
        if( crop_y+crop_h>src.extent(1)) {
          throw ParamOutOfBoundaryError("crop_y+crop_h", true, crop_y+crop_h, 
            src.extent(1) );
        }
      }
    
      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        // Crop the 2D array
        detail::cropNoCheck(src_slice, dst_slice, crop_x, crop_y, crop_w,
          crop_h, zero_out);
      }
    }

    template<typename T, int N>
    void cropAroundCenter(const blitz::Array<T,N>& src, blitz::Array<T,N>& dst,
	      const int eyes_distance)
    {
	    const int center_h = src.extent(N - 2);
	    const int center_w = src.extent(N - 1);

      const int crop_h = dst.extent(N - 2);
      const int crop_w = dst.extent(N - 1);

      // From Cosmin code
      const double D_EYES = 10.0;
      const double Y_UPPER = 5.0;
      const double model_size = 2 * D_EYES;
  
      const double ratio = eyes_distance / D_EYES;
      
      const double x0 = center_w * 0.5 * model_size;
      const double y0 = center_h - ratio * Y_UPPER;

/*	    const int crop_y   = center_h - crop_h / 2;
	    const int crop_x   = center_w - crop_w / 2;

*/
	    crop(src, dst, y0, x0, crop_h, crop_w);
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_CROP_H */
