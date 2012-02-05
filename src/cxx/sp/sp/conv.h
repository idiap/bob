/**
 * @file cxx/sp/sp/conv.h
 * @date Thu Feb 3 16:39:25 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based convolution product with zero padding
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_SP_CONV_H
#define BOB_SP_CONV_H

#include "core/Exception.h"
#include "core/array_assert.h"
#include "core/array_copy.h"
#include "core/array_index.h"
#include <algorithm>
#include <blitz/array.h>

namespace bob {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    /**
     * @brief Enumerations of the possible output size options
     */
    namespace Conv {
      enum SizeOption {
        Full,
        Same,
        Valid
      };
    }

    namespace detail {
      template <typename T>
      void convInternal(const blitz::Array<T,1> a, const blitz::Array<T,1> b, 
        blitz::Array<T,1> c, const int offset_0, const int offset_1)
      {
        const int M = a.extent(0);
        const int P = c.extent(0);

        int bl_offset = 0;
        int bu_offset = offset_1-1;
        int al_offset = 0;
        for(int i=0; i<P; ++i)
        {
          blitz::Array<T,1> b_s = b(blitz::Range(bu_offset,bl_offset,-1));
          blitz::Array<T,1> a_s = a(blitz::Range(al_offset,al_offset+bu_offset-bl_offset));
          c(i) = blitz::sum(a_s * b_s);
          if(i < offset_0) ++bu_offset;
          else ++al_offset;
          if(i >= M-offset_1) ++bl_offset;
        }
      }                                    

      template <typename T>
      void convInternal(const blitz::Array<T,2> A, const blitz::Array<T,2> B, 
        blitz::Array<T,2> C, const int offset0_0, const int offset0_1, 
        const int offset1_0, const int offset1_1)
      {
        const int M0 = A.extent(0);
        const int M1 = A.extent(1);
        const int P0 = C.extent(0);
        const int P1 = C.extent(1);

        int bl_offset0 = 0;
        int bu_offset0 = offset0_1-1;
        int al_offset0 = 0;

        int bl_offset1;
        int bu_offset1;
        int al_offset1;
        for(int i=0; i<P0; ++i)
        {
          blitz::Range r_b0(bu_offset0,bl_offset0,-1);
          blitz::Range r_a0(al_offset0,al_offset0+bu_offset0-bl_offset0);
          bl_offset1 = 0;
          bu_offset1 = offset1_1-1;
          al_offset1 = 0;

          for(int j=0; j<P1; ++j)
          {
            blitz::Array<T,2> B_s = B(r_b0,blitz::Range(bu_offset1,bl_offset1,-1));
            blitz::Array<T,2> A_s = A(r_a0,blitz::Range(al_offset1,al_offset1+bu_offset1-bl_offset1));
            C(i,j) = blitz::sum(A_s * B_s);
            if(j < offset1_0) ++bu_offset1;
            else ++al_offset1;
            if(j >= M1-offset1_1) ++bl_offset1;
          }
          if(i < offset0_0) ++bu_offset0;
          else ++al_offset0;
          if(i >= M0-offset0_1) ++bl_offset0;
        }
      }


      template<typename T> void convSep(const blitz::Array<T,2>& A, 
        const blitz::Array<T,1>& B, blitz::Array<T,2>& C,
        const enum Conv::SizeOption size_opt = Conv::Full)
      {
        for(int i=0; i<A.extent(1); ++i)
        {
          const blitz::Array<T,1> Arow = A(blitz::Range::all(), i);
          blitz::Array<T,1> Crow = C(blitz::Range::all(), i);
          conv(Arow, B, Crow, size_opt);
        }
      }

     template<typename T> void convSep(const blitz::Array<T,3>& A, 
        const blitz::Array<T,1>& B, blitz::Array<T,3>& C,
        const enum Conv::SizeOption size_opt = Conv::Full)
      {
        for(int i=0; i<A.extent(1); ++i)
          for(int j=0; j<A.extent(2); ++j)
          {
            const blitz::Array<T,1> Arow = A(blitz::Range::all(), i, j);
            blitz::Array<T,1> Crow = C(blitz::Range::all(), i, j);
            conv(Arow, B, Crow, size_opt);
          }
      }

      template<typename T> void convSep(const blitz::Array<T,4>& A, 
        const blitz::Array<T,1>& B, blitz::Array<T,4>& C,
        const enum Conv::SizeOption size_opt = Conv::Full)
      {
        for(int i=0; i<A.extent(1); ++i)
          for(int j=0; j<A.extent(2); ++j)
            for(int k=0; k<A.extent(3); ++k)
            {
              const blitz::Array<T,1> Arow = A(blitz::Range::all(), i, j, k);
              blitz::Array<T,1> Crow = C(blitz::Range::all(), i, j, k);
              conv(Arow, B, Crow, size_opt);
            }
      }
    }
 

    /**
     * @brief Gets the required size of the output of a 1D convolution product
     * @param a The size of the first input array
     * @param b The size of the second input array
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between a and b
     *                   * Valid: valid (part without padding)
     * @return The size of the output array
     */
    inline const int getConvOutputSize(const int a, const int b,
      const enum Conv::SizeOption size_opt = Conv::Full)
    {
      int res=0;
      // Size of "A + B - 1"
      if( size_opt == Conv::Full )
        res = std::max(0, a + b - 1);
      // Same size as A
      else if( size_opt == Conv::Same )
        res = a;
      // Size when not allowing any padding
      else 
        res = std::max(0, a - b + 1); 
      return res;
    }

    /**
     * @brief Gets the required size of the output of the convolution product
     * @param a The first input array a
     * @param b The second input array b
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between a and b
     *                   * Valid: valid (part without padding)
     * @return The size of the output array
     */
    template<typename T> 
    const blitz::TinyVector<int,1> getConvOutputSize(
      const blitz::Array<T,1>& a, const blitz::Array<T,1>& b,
      const enum Conv::SizeOption size_opt = Conv::Full)
    {
      blitz::TinyVector<int,1> size;
      size(0) = getConvOutputSize(a.extent(0), b.extent(0), size_opt);
      return size;
    }

    /**
     * @brief Gets the required size of the output of the convolution product
     * @param A The first input array A
     * @param B The second input array B
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between A and B
     *                   * Valid: valid (part without padding)
     * @return Size of the output
     */
    template<typename T> 
    const blitz::TinyVector<int,2> getConvOutputSize(
      const blitz::Array<T,2>& A, const blitz::Array<T,2>& B,
      const enum Conv::SizeOption size_opt = Conv::Full)
    {
      blitz::TinyVector<int,2> size;
      size(0) = 0;
      size(1) = 0;

      if( size_opt == Conv::Full )
      {
        size(0) = std::max(0, A.extent(0) + B.extent(0) - 1);
        size(1) = std::max(0, A.extent(1) + B.extent(1) - 1);
      }
      else if( size_opt == Conv::Same )
      {
        size(0) = A.extent(0);
        size(1) = A.extent(1);
      }
      else if( size_opt == Conv::Valid )
      {
        size(0) = std::max(0, A.extent(0) - B.extent(0) + 1);
        size(1) = std::max(0, A.extent(1) - B.extent(1) + 1);
      }

      return size;
    }

    /**
     * @brief Gets the required size of the output of the separable convolution product
     *        (Convolution of a X-D signal with a 1D kernel)
     * @param A The first input array A
     * @param B The second input array B
     * @param dim The dimension along which to convolve
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between A and B
     *                   * Valid: valid (part without padding)
     * @return Size of the output
     */
    template<typename T, int N> 
    const blitz::TinyVector<int,N> getConvSepOutputSize(const blitz::Array<T,N>& A,
      const blitz::Array<T,1>& B, const int dim,
      const enum Conv::SizeOption size_opt = Conv::Full)
    {
      blitz::TinyVector<int,N> res;
      res = A.shape();
      if(dim<N)
      {
        int a_size_d = A.extent(dim);
        int b_size = B.extent(0);
        res(dim) = getConvOutputSize(a_size_d, b_size, size_opt);
      }
      else 
      {
        throw bob::core::Exception();
      }
      return res;
    }





    /**
     * @brief 1D convolution of blitz arrays: c=a*b
     * @param a The first input array a
     * @param b The second input array b
     * @param c The output array c=a*b
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between A and B
     *                   * Valid: valid (part without padding)
     * @warning The output c should have the correct size
     */
    template <typename T>
    void conv(const blitz::Array<T,1> a, const blitz::Array<T,1> b, 
      blitz::Array<T,1> c, const enum Conv::SizeOption size_opt = Conv::Full)
    {
      const int N = b.extent(0);

      if(size_opt == Conv::Full)
        detail::convInternal(a, b, c, N-1, 1);
      else if(size_opt == Conv::Same)
        detail::convInternal(a, b, c, N/2, (N+1)/2);
      else
        detail::convInternal(a, b, c, 0, N);
    }

    /**
     * @brief 2D convolution of blitz arrays: C=A*B
     * @param A The first input array A
     * @param B The second input array B
     * @param C The output array C=A*B
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between A and B
     *                   * Valid: valid (part without padding)
     * @warning The output C should have the correct size
     */
    template <typename T>
    void conv(const blitz::Array<T,2> A, const blitz::Array<T,2> B, 
      blitz::Array<T,2> C, const enum Conv::SizeOption size_opt = Conv::Full)
    {
      const int N0 = B.extent(0);
      const int N1 = B.extent(1);

      if(size_opt == Conv::Full)
        detail::convInternal(A, B, C, N0-1, 1, N1-1, 1);
      else if(size_opt == Conv::Same)
        detail::convInternal(A, B, C, N0/2, (N0+1)/2, N1/2, (N1+1)/2);
      else
        detail::convInternal(A, B, C, 0, N0, 0, N1);
    }

    /**
     * @brief Convolution of a X-D signal with a 1D kernel (for separable convolution)
     *        along the specified dimension (C=A*B)
     * @param A The first input array A
     * @param B The second input array B
     * @param C The output array C=A*B along the dimension d (0 or 1)
     * @param dim The dimension along which to convolve
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between A and B
     *                   * Valid: valid (part without padding)
     */
    template<typename T, int N> void convSep(const blitz::Array<T,N>& A, 
      const blitz::Array<T,1>& B, blitz::Array<T,N>& C, const int dim,
      const enum Conv::SizeOption size_opt = Conv::Full)
    {
      // Gets the expected size for the results
      const blitz::TinyVector<int,N> Csize = getConvSepOutputSize(A, B, dim, size_opt);

      // Checks that C has the correct size and is zero base
      bob::core::array::assertSameShape(C, Csize);
      bob::core::array::assertZeroBase(C);
      // Checks that A and B are zero base
      bob::core::array::assertZeroBase(A);
      bob::core::array::assertZeroBase(B);

      if(dim==0)
        detail::convSep( A, B, C, size_opt);
      else if(dim<N)
      {
        const blitz::Array<T,N> Ap = (bob::core::array::ccopy(A)).transpose(dim,0);
        blitz::Array<T,N> Cp = C.transpose(dim,0);
        detail::convSep( Ap, B, Cp, size_opt);
      }
      else
        throw bob::core::Exception();
    }


    namespace deprecated {
      template<typename T> void convolveMirrorSep(const blitz::Array<T,2>& B, 
        const blitz::Array<T,1>& C, blitz::Array<T,2>& A, const int dim,
        const enum Conv::SizeOption size_opt = Conv::Full)
      {
        // Gets the expected size for the results
        const blitz::TinyVector<int,2> Asize = getConvSepOutputSize(B, C, dim, size_opt);

        // Checks that A has the correct size and is zero base
        bob::core::array::assertSameShape(A, Asize);
        bob::core::array::assertZeroBase(A);
        // Checks that B and C are zero base
        bob::core::array::assertZeroBase(B);
        bob::core::array::assertZeroBase(C);

        if(dim==0)
          convolveMirrorSepInt( B, C, A, size_opt);
        else if(dim<2)
        {
          const blitz::Array<T,2> Bp = (bob::core::array::ccopy(B)).transpose(dim,0);
          blitz::Array<T,2> Ap = A.transpose(dim,0);
          convolveMirrorSepInt( Bp, C, Ap, size_opt);
        }
        else
          throw bob::core::Exception();
      }

      template<typename T> void convolveMirrorSepInt(const blitz::Array<T,2>& B, 
        const blitz::Array<T,1>& C, blitz::Array<T,2>& A,
        const enum Conv::SizeOption size_opt = Conv::Full)
      { 
        for(int i=0; i<B.extent(1); ++i)
        {
          const blitz::Array<T,1> Brow = B(blitz::Range::all(), i);
          blitz::Array<T,1> Arow = A(blitz::Range::all(), i);
          convolveMirror(Brow, C, Arow, size_opt);
        }
      }

      template<typename T> void convolveMirror(const blitz::Array<T,1>& B, 
        const blitz::Array<T,1>& C, blitz::Array<T,1>& A,
        const enum Conv::SizeOption size_opt = Conv::Full)
      {
        const int Bsize = B.extent(0);
        const int Csize = C.extent(0);

        // Gets the expected size for the results
        const blitz::TinyVector<int,1> Asize = getConvOutputSize(B, C, size_opt);

        // Checks that A has the correct size and is zero base
        bob::core::array::assertSameShape(A, Asize);
        bob::core::array::assertZeroBase(A);
        // Checks that B and C are zero base
        bob::core::array::assertZeroBase(B);
        bob::core::array::assertZeroBase(C);
      
        A = 0;
        for(int i=0; i < Asize(0); ++i)
        {
          int i_shifted=0;
          if( size_opt == Conv::Full )
            i_shifted = i;
          else if( size_opt == Conv::Same )
            i_shifted = i + Csize / 2;

          if( size_opt == Conv::Full || size_opt == Conv::Same )
          {
            for(int j=i_shifted-(Csize-1); j <= i_shifted; ++j)
              A(i) += B( bob::core::array::mirrorInRange(j,0,Bsize-1) ) * 
                C( bob::core::array::mirrorInRange(i_shifted-j,0,Csize-1) );
          }
          else if( size_opt == Conv::Valid )
          {
            // Interpolation is useless in this case
            blitz::Range jb( i, i + Csize - 1), jc( Csize - 1, 0, -1); 
            A(i) = blitz::sum(B(jb) * C(jc) );
          }
          // Should be impossible to reach this point
          else
            throw bob::core::Exception();
        }
      }

    }
 
  }
/**
 * @}
 */
}

#endif /* BOB_SP_CONV_H */
