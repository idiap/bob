/**
 * @file src/cxx/ip/ip/shift.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to shift a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_SHIFT_H
#define TORCH5SPRO_IP_SHIFT_H 1

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
      void shiftNoCheck2D(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
        const int shift_x, const int shift_y, const bool allow_out)
      {
        bool is_y_out;
        for( int y=0; y<dst.extent(0); ++y) {
          is_y_out = y+shift_y<0 || y+shift_y>=src.extent(0);
          for( int x=0; x<dst.extent(1); ++x) {
            if( is_y_out || x+shift_x<0 || x+shift_x>=src.extent(1) )
              dst(y,x) = 0;
            else
              dst(y,x) = src( y + shift_y + src.lbound(0), 
                              x + shift_x + src.lbound(1) );
          }
        }
      }

    }


    /**
     * @brief Function which shifts a 2D blitz::array/image of a given type.
     *   The first dimension is the height (y-axis), whereas the second
     *   one is the width (x-axis).
     *   Pixels which are not in common with the input array are set to 0.
     * @param src The input blitz array
     * @param dst The output blitz array
     * @param shift_x The x-offset of the top left corner of the shifted area 
     * wrt. to the x-index of the top left corner of the blitz::array.
     * @param shift_y The y-offset of the top left corner of the shifted area
     * wrt. to the y-index of the top left corner of the blitz::array.
     * @param allow_out Whether an exception should be raised or not if the 
     * shifted blitz::array has no pixel in common with the input blitz::array.
     */
    template<typename T>
    void shift(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int shift_x, const int shift_y, const bool allow_out = false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
        dst.resize( src.extent(0), src.extent(1) );

      // Check parameters and throw exception if required
      if( (shift_x < -src.extent(1) || shift_x > src.extent(1) || 
          shift_y < -src.extent(0) || shift_y > src.extent(0)) && 
          !allow_out) {
        throw Torch::core::Exception();
      }
    
      // Shift the 2D array
      detail::shiftNoCheck2D<T>(src, dst, shift_x, shift_y, allow_out);
    }


    /**
     * @brief Function which shifts a 3D blitz::array/image of a given type.
     *   The first dimension is the number of planes, the second one the 
     *   height (y-axis), whereas the third one is the width (x-axis).
     *   Pixels which are not in common with the input array are set to 0.
     * @param src The input blitz array
     * @param dst The output blitz array
     * @param shift_x The x-offset of the top left corner of the shifted area 
     * wrt. to the x-index of the top left corner of the blitz::array.
     * @param shift_y The y-offset of the top left corner of the shifted area
     * wrt. to the y-index of the top left corner of the blitz::array.
     * @param allow_out Whether an exception should be raised or not if the 
     * shifted blitz::array has no pixel in common with the input blitz::array.
     */
    template<typename T>
    void shift(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst, 
      const int shift_x, const int shift_y, const bool allow_out = false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) ||
        dst.extent(2) != src.extent(2) )
        dst.resize( src.extent(0), src.extent(1), src.extent(2) );

      // Check parameters and throw exception if required
      if( (shift_x < -src.extent(2) || shift_x > src.extent(2) || 
          shift_y < -src.extent(1) || shift_y > src.extent(1)) && 
          !allow_out) {
        throw Torch::core::Exception();
      }
    
      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        // Shift the 2D array
        detail::shiftNoCheck2D(src_slice, dst_slice, shift_x, shift_y, 
          allow_out);
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_SHIFT_H */

