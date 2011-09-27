/**
 * @file src/cxx/sp/sp/extrapolate.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements various extrapolation techniques for 1D and 2D blitz
 * arrays.
 */

#ifndef TORCH5SPRO_SP_EXTRAPOLATE_H
#define TORCH5SPRO_SP_EXTRAPOLATE_H

#include <blitz/array.h>
#include "core/Exception.h"
#include "core/array_assert.h"
#include "core/array_index.h"

namespace Torch {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    /**
      * @brief Extrapolates a 1D array, padding with a constant
      */
    template<typename T>
    void extrapolateConstant(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst, 
      const T value)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0))
        throw Torch::core::Exception();

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
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      if(src.extent(0) > dst.extent(0) || src.extent(1) > dst.extent(1))
        throw Torch::core::Exception();

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
      Torch::sp::extrapolateConstant(src, dst, zero);
    }

    /**
      * @brief Extrapolates a 2D array, using zero padding
      */
    template<typename T>
    void extrapolateZero(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Calls extrapolate with the constant set to 0
      T zero = 0;
      Torch::sp::extrapolateConstant(src, dst, zero);
    }


    /**
      * @brief Extrapolates a 1D array, using nearest neighbour
      */
    template<typename T>
    void extrapolateNearest(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset = (dst.extent(0) - src.extent(0)) / 2;
      for(int i=0; i<dst.extent(0); ++i)
        dst(i) = src( Torch::core::array::keepInRange(i-offset, 0, src.extent(0)-1));
    }

    /**
      * @brief Extrapolates a 2D array, using nearest neighbour
      */
    template<typename T>
    void extrapolateNearest(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;
      for(int j=0; j<dst.extent(0); ++j)
        for(int i=0; i<dst.extent(1); ++i)
          dst(j,i) = src( Torch::core::array::keepInRange(j-offset_y, 0, src.extent(0)-1),
                          Torch::core::array::keepInRange(i-offset_x, 0, src.extent(1)-1));
    }
    

    /**
      * @brief Extrapolates a 1D array, using circular extrapolation
      */
    template<typename T>
    void extrapolateCircular(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

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
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

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
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset = (dst.extent(0) - src.extent(0)) / 2;

      // Sets left values
      for(int i=0; i<dst.extent(0); ++i)
        dst(i) = src( Torch::core::array::mirrorInRange(i-offset, 0, src.extent(0)-1));
    }

    /**
      * @brief Extrapolates a 2D array, using mirroring
      */
    template<typename T>
    void extrapolateMirror(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      // Computes offsets
      int offset_y = (dst.extent(0) - src.extent(0)) / 2;
      int offset_x = (dst.extent(1) - src.extent(1)) / 2;

      // Sets left values
      for(int j=0; j<dst.extent(0); ++j)
        for(int i=0; i<dst.extent(0); ++i)
          dst(j,i) = src( Torch::core::array::mirrorInRange(j-offset_y, 0, src.extent(0)-1),
                          Torch::core::array::mirrorInRange(i-offset_x, 0, src.extent(1)-1));
    }
 
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_INTERPOLATE_H */
