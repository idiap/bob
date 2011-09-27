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

    template<typename T>
    void extrapolateZero(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Calls extrapolate with the constant set to 0
      T zero = 0;
      Torch::sp::extrapolateConstant(src, dst, zero);
    }

    template<typename T>
    void extrapolateNearest(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      // Determines boundary values
      T left = src(src.lbound(0));
      T right = src(src.ubound(0));
      // Computes offsets
      int l_offset = (dst.extent(0) - src.extent(0)) / 2;
      int r_offset = l_offset + src.extent(0); 
      // Sets left values
      blitz::Range dst_range_l(0, l_offset-1);
      blitz::Array<T,1> dst_slice_l = dst(dst_range_l);
      dst_slice_l = left;
      // Sets middle values
      blitz::Range dst_range_m(l_offset, r_offset-1);
      blitz::Array<T,1> dst_slice_m = dst(dst_range_m);
      dst_slice_m = src;
      // Sets right values
      blitz::Range dst_range_r(r_offset, dst.extent(0)-1);
      blitz::Array<T,1> dst_slice_r = dst(dst_range_r);
      dst_slice_r = right;
    }

    template<typename T>
    void extrapolateCircular(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      // Computes offsets
      int l_offset = (dst.extent(0) - src.extent(0)) / 2;
      int r_offset = l_offset + src.extent(0);
      // Sets left values
      int s = src.extent(0);
      for(int i=0; i<l_offset; ++i)
        dst(i) = src( ((i-l_offset % s) + s) % s );
      // Sets middle values
      blitz::Range dst_range_m(l_offset, r_offset-1);
      blitz::Array<T,1> dst_slice_m = dst(dst_range_m);
      dst_slice_m = src;
      // Sets right values
      for(int i=r_offset; i<dst.extent(0); ++i)
        dst(i) = src( ((i-r_offset % s) + s) % s );
    }

    template<typename T>
    void extrapolateMirror(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      // Checks zero base
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);

      // Computes offsets
      int l_offset = (dst.extent(0) - src.extent(0)) / 2;
      int r_offset = l_offset + src.extent(0); 

      // Sets left values
      for(int i=0; i<l_offset; ++i)
        dst(i) = src( Torch::core::array::mirrorInRange(i-l_offset, 0, src.extent(0)-1));
      // Sets middle values
      blitz::Range dst_range_m(l_offset, r_offset-1);
      blitz::Array<T,1> dst_slice_m = dst(dst_range_m);
      dst_slice_m = src;
      // Sets right values
      for(int i=r_offset; i<dst.extent(0); ++i)
        dst(i) = src( Torch::core::array::mirrorInRange(i-l_offset, 0, src.extent(0)-1));
    }
 
 
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_INTERPOLATE_H */
