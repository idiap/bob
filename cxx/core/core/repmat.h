/**
 * @file cxx/core/core/repmat.h
 * @date Fri Jul 15 18:50:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions which allow to replicate
 * (matlab repmat-like) a 2D (or 1D) blitz array of a given type.
 * The output should be allocated and sized by the user.
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

#ifndef BOB_CORE_REPMAT_H
#define BOB_CORE_REPMAT_H

#include <limits>
#include <blitz/array.h>
#include "core/repmat_exception.h"
#include "core/array_assert.h"

namespace bob {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     * @brief Function which replicates an input 2D array like the matlab
     * repmat function.
     * 
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T> 
    void repmat_(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      int m = dst.extent(0) / src.extent(0);
      int n = dst.extent(1) / src.extent(1);
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
     * @brief Function which replicates an input 2D array like the matlab
     * repmat function.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that the
     * input and output matrices conform, use the repmat_() variant.
     */
    template<typename T> 
    void repmat(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      if(dst.extent(0) % src.extent(0) != 0)
        throw bob::core::RepmatNonMultipleLength(src.extent(0), dst.extent(0));
      if(dst.extent(1) % src.extent(1) != 0)
        throw bob::core::RepmatNonMultipleLength(src.extent(1), dst.extent(1));
      repmat_(src, dst);
    }

    /**
     * @brief Function which replicates an input 1D array, and generates a 2D 
     * array like the matlab repmat function.
     *
     * @param src Input array
     * @param dst Output array
     * @param row_vector_src Indicates whether the vector is considered as a
     *   row or as a column.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T> 
    void repmat_(const blitz::Array<T,1>& src, blitz::Array<T,2>& dst, 
      bool row_vector_src=false)
    {
      if(row_vector_src)
      {
        blitz::Array<T,2> dst_t = dst.transpose(1,0);
        repmat_(src, dst_t, false);
      }
      else // src is a column vector
      {
        int m = dst.extent(0) / src.extent(0);
        int n = dst.extent(1);
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
     * @brief Function which replicates an input 1D array, and generates a 2D 
     * array like the matlab repmat function.
     *
     * @param src Input array
     * @param dst Output array
     * @param row_vector_src Indicates whether the vector is considered as a
     *   row or as a column.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that the
     * input and output matrices conform, use the repmat_() variant.
     */ 
    template<typename T> 
    void repmat(const blitz::Array<T,1>& src, blitz::Array<T,2>& dst, 
      bool row_vector_src=false)
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      // Check dst length
      if(row_vector_src)
      {
        if(dst.extent(1) % src.extent(0) != 0)
          throw bob::core::RepmatNonMultipleLength(src.extent(0), dst.extent(1));
      }
      else // src is a column vector
      {
        if(dst.extent(0) % src.extent(0) != 0)
          throw bob::core::RepmatNonMultipleLength(src.extent(0), dst.extent(0));
      }
      repmat_(src, dst, row_vector_src);
    }

    /**
     * @brief Function which replicates an input 1D array, generating a new 
     * (larger) 1D array.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T> 
    void repvec_(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      int m = dst.extent(0) / src.extent(0);
      for(int i=0; i<m; ++i)
      {
        blitz::Array<T,1> dst_m = 
          dst(blitz::Range(src.extent(0)*i,src.extent(0)*(i+1)-1));
        dst_m = src;
      }
    }


    /**
     * @brief Function which replicates an input 1D array, generating a new 
     * (larger) 1D array.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that the
     * input and output matrices conform, use the repmat_() variant.
     */
    template<typename T> 
    void repvec(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      if(dst.extent(0) % src.extent(0) != 0)
        throw bob::core::RepmatNonMultipleLength(src.extent(0), dst.extent(0));
      repvec_(src,dst);
    }

    /**
     * @brief Function which replicates the elements of an input 1D array, 
     * generating a new (larger) 1D array. In contrast to repvec, repetitions
     * occur at the element level, rather than at the vector level.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T> 
    void repelem_(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      int size_block = dst.extent(0) / src.extent(0);
      for(int i=0; i<src.extent(0); ++i)
      {
        blitz::Array<T,1> dst_m = 
          dst(blitz::Range(size_block*i, size_block*(i+1)-1));
        dst_m = src(i);
      }
    }


    /**
     * @brief Function which replicates the elements of an input 1D array, 
     * generating a new (larger) 1D array. In contrast to repvec, repetitions
     * occur at the element level, rather than at the vector level.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that the
     * input and output matrices conform, use the repmat_() variant.
     */
    template<typename T> 
    void repelem(const blitz::Array<T,1>& src, blitz::Array<T,1>& dst)
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      if(dst.extent(0) % src.extent(0) != 0)
        throw bob::core::RepmatNonMultipleLength(src.extent(0), dst.extent(0));
      repelem_(src,dst);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_CORE_REPMAT_H */
