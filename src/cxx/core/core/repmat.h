/**
 * @file src/cxx/core/core/repmat.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines functions which allow to replicate 
 * (matlab repmat-like) a 2D (or 1D) blitz array of a given type. 
 * The output should be allocated and sized by the user.
 * 
 */

#ifndef TORCH5SPRO_CORE_REPMAT_H
#define TORCH5SPRO_CORE_REPMAT_H

#include <limits>
#include <blitz/array.h>
#include "core/array_assert.h"

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     * @brief Function which replicates an input 2D array like the matlab
     * repmat function.
     */
    template<typename T> 
    void repmat(const blitz::Array<T,2>& src, int m, int n, 
      blitz::Array<T,2>& dst) 
    {
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);
      Torch::core::array::assertSameDimensionLength(dst.extent(0), m*src.extent(0));
      Torch::core::array::assertSameDimensionLength(dst.extent(1), n*src.extent(1));
      for(int i=0; i<m; ++i)
      {
        for(int j=0; j<n; ++j)
        {
          blitz::Array<T,2> dst_mn = 
            dst(blitz::Range(src.extent(0)*i,src.extent(0)*(i+1)-1), 
              blitz::Range(src.extent(1)*j,src.extent(1)*(j+1)-1));
          dst_mn = src;
        }
      }
    }

    /**
     * @brief Function which replicates an input 1D array, and generates a 2D 
     * array like the matlab repmat function.
     *
     * @param row_vector_src Indicates whether the vector is considered as a
     *   row or as a column.
     */
    template<typename T> 
    void repmat(const blitz::Array<T,1>& src, int m, int n, 
      blitz::Array<T,2>& dst, bool row_vector_src=false)
    {
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);
      if(row_vector_src)
      {
        Torch::core::array::assertSameDimensionLength(dst.extent(0), m);
        Torch::core::array::assertSameDimensionLength(dst.extent(1), n*src.extent(0));
        for(int i=0; i<m; ++i)
        {
          for(int j=0; j<n; ++j)
          {
            blitz::Array<T,1> dst_mn = 
              dst(i, blitz::Range(src.extent(0)*j,src.extent(0)*(j+1)-1));
            dst_mn = src;
          }
        }
      }
      else // src is a column vector
      {
        Torch::core::array::assertSameDimensionLength(dst.extent(0), m*src.extent(0));
        Torch::core::array::assertSameDimensionLength(dst.extent(1), n);
        for(int i=0; i<m; ++i)
        {
          for(int j=0; j<n; ++j)
          {
            blitz::Array<T,1> dst_mn = 
              dst(blitz::Range(src.extent(0)*i,src.extent(0)*(i+1)-1), j);
            dst_mn = src;
          }
        }
      }
    }

    /**
     * @brief Function which replicates an input 1D array, generating a new 
     * (larger) 1D array.
     */
    template<typename T> 
    void repvec(const blitz::Array<T,1>& src, int m, blitz::Array<T,1>& dst)
    {
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);
      Torch::core::array::assertSameDimensionLength(dst.extent(0), m*src.extent(0));
      for(int i=0; i<m; ++i)
      {
        blitz::Array<T,1> dst_m = 
          dst(blitz::Range(src.extent(0)*i,src.extent(0)*(i+1)-1));
        dst_m = src;
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_REPMAT_H */
