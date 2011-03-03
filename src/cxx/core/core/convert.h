/**
 * @file src/cxx/core/core/convert.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines functions which allows to convert/rescale a 
 * blitz array of a given type into a blitz array of an other type. Typically,
 * this can be used to rescale a 16 bit precision grayscale image (2d array)
 * into an 8 bit precision grayscale image.
 * 
 * @see Torch::core::cast
 */

#ifndef TORCH5SPRO_CORE_CONVERT_H
#define TORCH5SPRO_CORE_CONVERT_H 1

#include "core/logging.h"
#include "core/Exception.h"
#include <limits>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     * @brief Function which converts a 1D blitz::array of a given type into
     * a 1D blitz::array of an other type, using the given ranges.
     */
    template<typename T, typename U> 
    blitz::Array<T,1> convert(const blitz::Array<U,1>& src, 
      T dst_min, T dst_max, U src_min, U src_max) 
    {
      blitz::Array<T,1> dst( src.extent(0) );
      if( src_min == src_max)
        throw Torch::core::Exception();
      double src_ratio = 1. / ( src_max - src_min);
      T dst_diff = dst_max - dst_min;
      for( int i=0; i<src.extent(0); ++i) {
        if( src(i) < src_min || src(i) > src_max )
          throw Torch::core::Exception();
        // If the destination is an integer-like type, we need to add 0.5 s.t.
        // the round done by the implicit conversion is correct
        dst(i) = dst_min + (((src(i)-src_min)*src_ratio) * dst_diff + (std::numeric_limits<T>::is_integer?0.5:0));
      }
      return dst;
    }

    /**
     * @brief Function which converts a 2D blitz::array of a given type into
     * a 2D blitz::array of an other type, using the given ranges.
     */
    template<typename T, typename U> 
    blitz::Array<T,2> convert(const blitz::Array<U,2>& src, 
      T dst_min, T dst_max, U src_min, U src_max) 
    {
      blitz::Array<T,2> dst( src.extent(0), src.extent(1) );
      if( src_min == src_max)
        throw Torch::core::Exception();
      double src_ratio = 1. / ( src_max - src_min);
      T dst_diff = dst_max - dst_min;
      for( int i=0; i<src.extent(0); ++i) 
        for( int j=0; j<src.extent(1); ++j) {
          if( src(i,j) < src_min || src(i,j) > src_max )
            throw Torch::core::Exception();
          // If the destination is an integer-like type, we need to add 0.5 
          // s.t. the round done by the implicit conversion is correct
          dst(i,j) = dst_min + (((src(i,j)-src_min)*src_ratio) * dst_diff + 
            (std::numeric_limits<T>::is_integer?0.5:0));
        }
      return dst;
    }

    /**
     * @brief Function which converts a 3D blitz::array of a given type into
     * a 3D blitz::array of an other type, using the given ranges.
     */
    template<typename T, typename U> 
    blitz::Array<T,3> convert(const blitz::Array<U,3>& src, 
      T dst_min, T dst_max, U src_min, U src_max) 
    {
      blitz::Array<T,3> dst( src.extent(0), src.extent(1), src.extent(2) );
      if( src_min == src_max)
        throw Torch::core::Exception();
      double src_ratio = 1. / ( src_max - src_min);
      T dst_diff = dst_max - dst_min;
      for( int i=0; i<src.extent(0); ++i)
        for( int j=0; j<src.extent(1); ++j) 
          for( int k=0; k<src.extent(2); ++k) {
            if( src(i,j,k) < src_min || src(i,j,k) > src_max )
              throw Torch::core::Exception();
            // If the destination is an integer-like type, we need to add 0.5 
            // s.t. the round done by the implicit conversion is correct
            dst(i,j,k) = dst_min + (((src(i,j,k)-src_min)*src_ratio) * 
              dst_diff + (std::numeric_limits<T>::is_integer?0.5:0));
          }
      return dst;
    }

    /**
     * @brief Function which converts a 4D blitz::array of a given type into
     * a 4D blitz::array of an other type, using the given ranges.
     */
    template<typename T, typename U> 
    blitz::Array<T,4> convert(const blitz::Array<U,4>& src, 
      T dst_min, T dst_max, U src_min, U src_max) 
    {
      blitz::Array<T,4> dst( src.extent(0), src.extent(1), src.extent(2),
        src.extent(3) );
      if( src_min == src_max)
        throw Torch::core::Exception();
      double src_ratio = 1. / ( src_max - src_min);
      T dst_diff = dst_max - dst_min;
      for( int i=0; i<src.extent(0); ++i)
        for( int j=0; j<src.extent(1); ++j) 
          for( int k=0; k<src.extent(2); ++k)
            for( int l=0; l<src.extent(3); ++l) {
              if( src(i,j,k,l) < src_min || src(i,j,k,l) > src_max )
                throw Torch::core::Exception();
              // If the destination is an integer-like type, we need to add 0.5
              // s.t. the round done by the implicit conversion is correct
              dst(i,j,k,l) = dst_min + (((src(i,j,k,l)-src_min)*src_ratio) *
                dst_diff + (std::numeric_limits<T>::is_integer?0.5:0));
            }
      return dst;
    }


    /**
     * @brief Function which converts a blitz::array of a given type into
     * a blitz::array of an other type, using the full type range.
     */
    template<typename T, typename U, int d> 
    blitz::Array<T,d> convert(const blitz::Array<U,d>& src)
    {
      return convert<T,U>( src, std::numeric_limits<T>::min(), 
        std::numeric_limits<T>::max(), std::numeric_limits<U>::min(), 
        std::numeric_limits<U>::max() );
    }

    /**
     * @brief Function which converts a blitz::array of a given type into
     * a blitz::array of an other type, using the given range for the 
     * destination.
     */
    template<typename T, typename U, int d> 
    blitz::Array<T,d> convertToRange(const blitz::Array<U,d>& src, 
      T dst_min, T dst_max) 
    {
      return convert<T,U>( src, dst_min, dst_max, 
        std::numeric_limits<U>::min(), std::numeric_limits<U>::max() );
    }

    /**
     * @brief Function which converts a blitz::array of a given type into
     * a blitz::array of an other type, using the given range for the 
     * source.
     */
    template<typename T, typename U, int d> 
    blitz::Array<T,d> convertFromRange(const blitz::Array<U,d>& src, 
      U src_min, U src_max) 
    {
      return convert<T,U>( src, std::numeric_limits<T>::min(), 
        std::numeric_limits<T>::max(), src_min, src_max );
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_CAST_H */

