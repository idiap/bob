/**
 * @file src/cxx/ip/ip/shift.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to shift a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_SHIFT_H
#define TORCH5SPRO_IP_SHIFT_H

#include "core/array_index.h"
#include "ip/Exception.h"
#include "ip/common.h"

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
        * @brief Function which shifts a 2D blitz::array/image of a given type.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param shift_y The y-offset of the top left corner of the shifted 
        * area wrt. to the y-index of the top left corner of the blitz::array.
        * @param shift_x The x-offset of the top left corner of the shifted 
        * area wrt. to the x-index of the top left corner of the blitz::array.
        * @param zero_out Whether the shifted area which is out of the boundary
        * of the input blitz array should be filled with zero values, or with 
        * the intensity of the closest pixel in the neighbourhood.
        */
      template<typename T>
      void shiftNoCheck2D(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
        const int shift_y, const int shift_x, const bool zero_out)
      {
        bool is_y_out;
        int y_src, x_src;
        for( int y=0; y<dst.extent(0); ++y) {
          is_y_out = y+shift_y<0 || y+shift_y>=src.extent(0);
          y_src = tca::keepInRange( y+shift_y, 0, src.extent(0)-1);
          for( int x=0; x<dst.extent(1); ++x) {
            if( is_y_out || x+shift_x<0 || x+shift_x>=src.extent(1) ) {
              x_src = tca::keepInRange( x+shift_x, 0, src.extent(1)-1);
              dst(y,x) = (zero_out ? 0 : 
                src( y_src+src.lbound(0), x_src+src.lbound(1)) );
            }
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
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param shift_y The y-offset of the top left corner of the shifted area
      * wrt. to the y-index of the top left corner of the blitz::array.
      * @param shift_x The x-offset of the top left corner of the shifted area 
      * wrt. to the x-index of the top left corner of the blitz::array.
      * @param allow_out Whether an exception should be raised or not if the 
      * shifted blitz::array has no pixel in common with the input blitz::array.
      * @param zero_out Whether the shifted which is out of the boundary
      * of the input blitz array should be filled with zero values, or with 
      * the intensity of the closest pixel in the neighbourhood.
      */
    template<typename T>
    void shift(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int shift_y, const int shift_x, const bool allow_out = false,
      const bool zero_out = false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
        dst.resize( src.extent(0), src.extent(1) );
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }

      // Check parameters and throw exception if required
      if( (shift_x <= -src.extent(1) || shift_x >= src.extent(1) || 
          shift_y <= -src.extent(0) || shift_y >= src.extent(0)) && 
          !allow_out) 
      {
        if( shift_x <= -src.extent(1) ) {
          throw ParamOutOfBoundaryError("shift_x", false, shift_x, 
            -src.extent(1)+1);
        }
        else if( shift_y <= -src.extent(0) ) {
          throw ParamOutOfBoundaryError("shift_y", false, shift_y, 
            -src.extent(0)+1);
        }
        else if( shift_x >= src.extent(1) ) {
          throw ParamOutOfBoundaryError("shift_x", true, shift_x, 
            src.extent(1)-1);
        }
        else if( shift_y >= src.extent(0) ) {
          throw ParamOutOfBoundaryError("shift_y", true, shift_y, 
            src.extent(0)-1);
        }
        else
          throw Exception();
      }
    
      // Shift the 2D array
      detail::shiftNoCheck2D<T>(src, dst, shift_y, shift_x, zero_out);
    }


    /**
      * @brief Function which shifts a 3D blitz::array/image of a given type.
      *   The first dimension is the number of planes, the second one the 
      *   height (y-axis), whereas the third one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param shift_y The y-offset of the top left corner of the shifted area
      * wrt. to the y-index of the top left corner of the blitz::array.
      * @param shift_x The x-offset of the top left corner of the shifted area 
      * wrt. to the x-index of the top left corner of the blitz::array.
      * @param allow_out Whether an exception should be raised or not if the 
      * shifted blitz::array has no pixel in common with the input blitz::array.
      * @param zero_out Whether the shifted which is out of the boundary
      * of the input blitz array should be filled with zero values, or with 
      * the intensity of the closest pixel in the neighbourhood.
      */
    template<typename T>
    void shift(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst, 
      const int shift_y, const int shift_x, const bool allow_out = false,
      const bool zero_out = false)
    {
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) ||
        dst.extent(2) != src.extent(2) )
        dst.resize( src.extent(0), src.extent(1), src.extent(2) );
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 || dst.base(2) != 0 ) {
        const blitz::TinyVector<int,3> zero_base = 0;
        dst.reindexSelf( zero_base );
      }

      // Check parameters and throw exception if required
      if( (shift_x <= -src.extent(2) || shift_x >= src.extent(2) || 
          shift_y <= -src.extent(1) || shift_y >= src.extent(1)) && 
          !allow_out) 
      {
        if( shift_x <= -src.extent(2) ) {
          throw ParamOutOfBoundaryError("shift_x", false, shift_x, 
            -src.extent(2)+1);
        }
        else if( shift_y <= -src.extent(1) ) {
          throw ParamOutOfBoundaryError("shift_y", false, shift_y, 
            -src.extent(1)+1);
        }
        else if( shift_x >= src.extent(2) ) {
          throw ParamOutOfBoundaryError("shift_x", true, shift_x, 
            src.extent(2)-1);
        }
        else if( shift_y >= src.extent(1) ) {
          throw ParamOutOfBoundaryError("shift_y", true, shift_y, 
            src.extent(1)-1);
        }
        else
          throw Exception();
      }
    
      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        // Shift the 2D array
        detail::shiftNoCheck2D(src_slice, dst_slice, shift_y, shift_x, 
          zero_out);
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_SHIFT_H */
