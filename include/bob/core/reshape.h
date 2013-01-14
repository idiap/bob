/**
 * @file bob/core/reshape.h
 * @date Sun Jul 17 13:31:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions which allow to reshape
 * (matlab repshape-like) a 2D (or 1D) blitz array of a given type.
 * The output should be allocated and sized by the user.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_CORE_RESHAPE_H
#define BOB_CORE_RESHAPE_H

#include <limits>
#include <blitz/array.h>
#include "bob/core/array_assert.h"
#include "bob/core/reshape_exception.h"

namespace bob {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     * TODO: Allows to reshape arrays of any number of dimensions
     */

    /**
     * @brief Function which reshapes an input 2D array like the matlab
     * reshape function.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T> 
    void reshape_(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      int is=0;
      int js=0;
      for(int j=0; j<dst.extent(1); ++j)
        for(int i=0; i<dst.extent(0); ++i)
        {
          dst(i,j) = src(is,js);
          ++is;
          if(is>=src.extent(0)) 
          {
            ++js;
            is=0;
          }
        }
    }

    /**
     * @brief Function which reshapes an input 2D array like the matlab
     * reshape function.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that the
     * input and output matrices conform, use the reshape_() variant.
     */
    template<typename T> 
    void reshape(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      const int d0 = src.extent(0)*src.extent(1);
      const int d1 = dst.extent(0)*dst.extent(1);
      if(d0 != d1) throw bob::core::ReshapeDifferentNumberOfElements(d0, d1);
      reshape_(src, dst);
    }

    /**
     * @brief Function which reshapes an input 2D array like the matlab
     * reshape function, into a 1D array.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T> 
    void reshape_(const blitz::Array<T,2>& src, blitz::Array<T,1>& dst) 
    {
      int n_blocks = src.extent(1);
      int size_block = src.extent(0);
      for(int b=0; b<n_blocks; ++b)
      {
        blitz::Array<T,1> src_b = src(blitz::Range::all(), b);
        blitz::Array<T,1> dst_b = dst(blitz::Range(b*size_block, (b+1)*size_block-1));
        dst_b = src_b;
      }
    }

    /**
     * @brief Function which reshapes an input 2D array like the matlab
     * reshape function, into a 1D array.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that the
     * input and output matrices conform, use the reshape_() variant.
     */
    template<typename T> 
    void reshape(const blitz::Array<T,2>& src, blitz::Array<T,1>& dst) 
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      const int d0 = src.extent(0)*src.extent(1);
      const int d1 = dst.extent(0);
      if(d0 != d1) throw bob::core::ReshapeDifferentNumberOfElements(d0, d1);
      reshape_(src, dst);
    }

    /**
     * @brief Function which reshapes an input 1D array like the matlab
     * reshape function, into a 2D array.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     */
    template<typename T> 
    void reshape_(const blitz::Array<T,1>& src, blitz::Array<T,2>& dst)
    {
      int n_blocks = dst.extent(1);
      int size_block = dst.extent(0);
      for(int b=0; b<n_blocks; ++b)
      {
        blitz::Array<T,1> src_b = src(blitz::Range(b*size_block, (b+1)*size_block-1));
        blitz::Array<T,1> dst_b = dst(blitz::Range::all(), b);
        dst_b = src_b;
      }
    }

    /**
     * @brief Function which reshapes an input 1D array like the matlab
     * reshape function, into a 2D array.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that the
     * input and output matrices conform, use the reshape_() variant.
     */
    template<typename T> 
    void reshape(const blitz::Array<T,1>& src, blitz::Array<T,2>& dst) 
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      const int d0 = src.extent(0);
      const int d1 = dst.extent(0)*dst.extent(1);
      if(d0 != d1) throw bob::core::ReshapeDifferentNumberOfElements(d0, d1);
      reshape_(src, dst);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_CORE_RESHAPE_H */
