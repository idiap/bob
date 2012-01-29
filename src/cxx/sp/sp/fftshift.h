/**
 * @file cxx/sp/sp/fftshift.h
 * @date Thu Apr 14 13:39:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement blitz-based fftshift functions
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_SP_FFTSHIFT_H
#define BOB_SP_FFTSHIFT_H

#include <complex>
#include <blitz/array.h>
#include "core/array_assert.h"


namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp {

    namespace detail {

      /**
        * @brief Perform the fftshift operation without check the 
        *   input/outpout
        */
      template <typename T>
      void fftshiftNoCheck( const int center, const blitz::Array<T,1>& A, 
        blitz::Array<T,1>& B)
      {
        if(center>0)
        {
          // Process source left
          blitz::Range  A_d1(0, center-1),
                        B_d1(B.extent(0)-center, B.extent(0)-1);
          B(B_d1) = A(A_d1);
        }
        if(center<A.extent(0))
        {
         // Process source right
         blitz::Range  A_d2(center, A.extent(0)-1),
                       B_d2(0, B.extent(0)-center-1);
         B(B_d2) = A(A_d2);
        }
      }

      template <typename T>
      void fftshiftNoCheck( const int centerY, const int centerX, 
        const blitz::Array<T,2>& A, blitz::Array<T,2>& B)
      {
        if(centerY>0 && centerX>0)
        {
          // Process source quadrant 1 (Top left)
          blitz::Range  A_y1(0, centerY-1),
                        A_x1(0, centerX-1),
                        B_y1(B.extent(0)-centerY, B.extent(0)-1),
                        B_x1(B.extent(1)-centerX, B.extent(1)-1);
          B(B_y1,B_x1) = A(A_y1,A_x1);
        }
        if(centerY<A.extent(0) && centerX<A.extent(1))
        {
          // Process source quadrant 3 (Bottom right)
          blitz::Range  A_y3(centerY, A.extent(0)-1),
                        A_x3(centerX, A.extent(1)-1),
                        B_y3(0, B.extent(0)-centerY-1),
                        B_x3(0, B.extent(1)-centerX-1);
          B(B_y3,B_x3) = A(A_y3,A_x3);
        }
        if(centerY>0 && centerX<A.extent(1))
        {
          // Process source quadrant 2 (Top right)
          blitz::Range  A_y2(0, centerY-1),
                        A_x2(centerX, A.extent(1)-1),
                        B_y2(B.extent(0)-centerY, B.extent(0)-1),
                        B_x2(0, B.extent(1)-centerX-1);
          B(B_y2,B_x2) = A(A_y2,A_x2);
        }
        if(centerY<A.extent(0) && centerX>0)
        {
          // Process source quadrant 4 (Bottom left)
          blitz::Range  A_y4(centerY, A.extent(0)-1),
                        A_x4(0, centerX-1),
                        B_y4(0, B.extent(0)-centerY-1),
                        B_x4(B.extent(1)-centerX, B.extent(1)-1);
          B(B_y4,B_x4) = A(A_y4,A_x4);
        }
      }
    }

    /**
     * @brief Rearrange the output of the fft by moving the zero-frequency 
     * component to the center of the 1D blitz array. 
     */
    template <typename T>
    void fftshift(const blitz::Array<T,1>& A, blitz::Array<T,1>& B)
    {
      // Check input
      bob::core::array::assertZeroBase(A);
      // Check output
      bob::core::array::assertZeroBase(B);
      bob::core::array::assertSameShape(A,B);

      // Center location
      int size_by_2 = A.extent(0)/2 + (A.extent(0) % 2);
    
      detail::fftshiftNoCheck(size_by_2, A, B);
    }

    /**
     * @brief Swap the two halves of the 1D blitz array wrt. to its center
     * ifftshift reverts the result of fftshift, which is important for 
     * dimensions of odd size.
     */
    template <typename T>
    void ifftshift(const blitz::Array<T,1>& A, blitz::Array<T,1>& B)
    {
      // Check input
      bob::core::array::assertZeroBase(A);
      // Check output
      bob::core::array::assertZeroBase(B);
      bob::core::array::assertSameShape(A,B);

      // Center location
      int size_by_2 = A.extent(0)/2;
    
      detail::fftshiftNoCheck(size_by_2, A, B);
    }

    /**
     * @brief Rearrange the output of the fft by moving the zero-frequency 
     * component to the center of the 2D blitz array. Therefore, it swaps 
     * the first quadrant with the third and the second quadrant with the 
     * fourth.
     */
    template <typename T>
    void fftshift(const blitz::Array<T,2>& A, blitz::Array<T,2>& B)
    {
      // Check input
      bob::core::array::assertZeroBase(A);
      // Check output
      bob::core::array::assertZeroBase(B);
      bob::core::array::assertSameShape(A,B);

      // Center location
      int h_by_2 = A.extent(0)/2 + (A.extent(0) % 2);
      int w_by_2 = A.extent(1)/2 + (A.extent(1) % 2); 
    
      detail::fftshiftNoCheck(h_by_2, w_by_2, A, B);
    }

    /**
     * @brief Swap the first quadrant with the third and the second quadrant 
     * with the fourth. ifftshift reverts the result of fftshift, which is 
     * important for dimensions of odd size.
     */
    template <typename T>
    void ifftshift(const blitz::Array<T,2>& A, blitz::Array<T,2>& B)
    {
      // Check input
      bob::core::array::assertZeroBase(A);
      // Check output
      bob::core::array::assertZeroBase(B);
      bob::core::array::assertSameShape(A,B);

      // Center location
      int h_by_2 = A.extent(0)/2;
      int w_by_2 = A.extent(1)/2; 

      detail::fftshiftNoCheck(h_by_2, w_by_2, A, B);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_SP_FFTSHIFT_H */
