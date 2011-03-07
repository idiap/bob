/**
 * @file src/cxx/ip/ip/crop.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to crop a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_CROP_H
#define TORCH5SPRO_IP_CROP_H 1

#include "core/logging.h"
#include "core/Exception.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {

      template<typename T>
      void cropNoCheck2D(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
        const int crop_x, const int crop_y, const int crop_w, const int crop_h,
        const bool allow_out)
      {
        bool is_y_out;
        for( int y=0; y<crop_h; ++y) {
          is_y_out = y+crop_y<0 || y+crop_y>=src.extent(0);
          for( int x=0; x<crop_w; ++x) {
            if( is_y_out || x+crop_x<0 || x+crop_x>=src.extent(1) )
              dst(y,x) = 0;
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
     * @param allow_out Whether an exception should be raised or not if a part
     * of the cropping area is out of the boundary of the input blitz array.
     * If not, pixels cropped out of the input blitz array have zero values. 
     */
    template<typename T>
    void crop(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int crop_x, const int crop_y, const int crop_w, const int crop_h,
      const bool allow_out=false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != crop_h || dst.extent(1) != crop_w )
        dst.resize( crop_h, crop_w );

      // Check parameters and throw exception if required
      if( (crop_x<0 || crop_y<0 || crop_w<0 || crop_h<0 || 
        crop_x+crop_w>src.extent(1) || crop_y+crop_h>src.extent(0) ) &&
        !allow_out) {
        throw Torch::core::Exception();
      }
    
      // Crop the 2D array
      detail::cropNoCheck2D<T>(src, dst, crop_x, crop_y, crop_w, crop_h, allow_out);
    }


    /**
     * @brief Function which crops a 3D blitz::array/image of a given type.
     *   The first dimension is the number of planes, the second one the 
     *   height (y-axis), whereas the third one is the width (x-axis).
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
     * If not, pixels cropped out of the input blitz array have zero values. 
     */
    template<typename T>
    void crop(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst, 
      const int crop_x, const int crop_y, const int crop_w, const int crop_h,
      const bool allow_out=false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != crop_h || 
          dst.extent(2) != crop_w )
        dst.resize( src.extent(0), crop_h, crop_w );

      // Check parameters and throw exception if required
      if( (crop_x<0 || crop_y<0 || crop_w<0 || crop_h<0 || 
        crop_x+crop_w>src.extent(2) || crop_y+crop_h>src.extent(1) ) &&
        !allow_out) {
        throw Torch::core::Exception();
      }
    
      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        // Crop the 2D array
        detail::cropNoCheck2D(src_slice, dst_slice, crop_x, crop_y, crop_w,
          crop_h, allow_out);
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_CROP_H */

