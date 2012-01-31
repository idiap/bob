/**
 * @file cxx/ip/ip/zigzag.h
 * @date Tue Apr 5 17:28:28 2011 +0200
 * @author Niklas Johansson <niklas.johansson@idiap.ch>
 *
 * @brief This file defines a function to extract a 1D zigzag pattern from
 * 2D dimensional array as described in:
 *   "Polynomial Features for Robust Face Authentication",
 *   from C. Sanderson and K. Paliwal, in the proceedings of the
 *   IEEE International Conference on Image Processing 2002.
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

#ifndef BOB5SPRO_IP_ZIGZAG_H
#define BOB5SPRO_IP_ZIGZAG_H

#include "core/array_assert.h"
#include "ip/Exception.h"

namespace tca = bob::core::array;

namespace bob {
	/**
	 * \ingroup libip_api
	 * @{
	 *
	 */
	namespace ip {

    namespace detail {
      /**
       * @brief Extract the zigzag pattern from a 2D blitz::array, as
       * described in:
       *   "Polynomial Features for Robust Face Authentication", 
       *   from C. Sanderson and K. Paliwal, in the proceedings of the 
       *   IEEE International Conference on Image Processing 2002.
       * @param src The input blitz array
       * @param dst The output blitz array
       * @param right_first Set to true to reverse the initial zigzag order. 
       *   By default, the direction is left-to-right for the first diagonal.
       */
      template<typename T>
      void zigzagNoCheck(const blitz::Array<T,2>& src, blitz::Array<T,1>& dst, 
        const bool right_first)
      {
        // Number of coefficients to keep
        const int n_coef_kept = dst.extent(0);

        // Define constants
        const int min_dim = std::min(src.extent(0), src.extent(1));
        const int max_dim = std::max(src.extent(0), src.extent(1));
        // Index of the current diagonal
        int current_diagonal = 0;
        // Direction of the current diagonal
        int diagonal_left_to_right_p = !right_first;
        // Offset the point in the current diagonal from its origin
        int diagonal_offset = 0;
        // Length of the current diagonal
        int diagonal_length = 1;

        // Get all required coefficients 
        for(int ind=0; ind < n_coef_kept; ++ind ) {
          int x, y;
       
          // Conditions used to determine the coordinates (x,y) in the 2D 
          // array given the 1D index in the zigzag
          if(diagonal_left_to_right_p) {
            if( current_diagonal>src.extent(0)-1 ) {
              x = current_diagonal-(src.extent(0)-1) + diagonal_offset;
              y = (src.extent(0)-1) - diagonal_offset;
            }
            else {
              x = diagonal_offset;
              y = current_diagonal - diagonal_offset;
            }
          } else {
            if( current_diagonal>src.extent(1)-1 ) {
              x = (src.extent(1)-1) - diagonal_offset;
              y = current_diagonal-(src.extent(1)-1) + diagonal_offset;
            }
            else {
              x = current_diagonal - diagonal_offset;
              y = diagonal_offset;
            }
          }

          // save the value in the 1D array
          dst(ind) = src(y, x);

          // Increment the diagonal offset
          ++diagonal_offset;
          // Update information about the current diagonal if required
          if(diagonal_length <= diagonal_offset) {
            // Increment current diagonal
            ++current_diagonal;
            // Reverse the direction in the diagonal
            diagonal_left_to_right_p = !diagonal_left_to_right_p; 
            // Reset the offset in the diagonal to 0
            diagonal_offset = 0; 
            // Determine the new size of the diagonal
            if( current_diagonal<min_dim )
              ++diagonal_length;
            else if( current_diagonal>=max_dim)
              --diagonal_length;
          }
        }
      }
    }

		/**
		 * @brief Extract the zigzag pattern from a 2D blitz::array, as described 
     * in:
     *   "Polynomial Features for Robust Face Authentication", 
     *   from C. Sanderson and K. Paliwal, in the proceedings of the 
     *   IEEE International Conference on Image Processing 2002.
		 * @param src The input blitz array
		 * @param dst The output blitz array
		 * @param n_coef_kept The number of coefficients to be kept
		 * @param right_first Set to true to reverse the initial zigzag order. 
     *   By default, the direction is left-to-right for the first diagonal.
		 */
		template<typename T>
    void zigzag(const blitz::Array<T,2>& src, blitz::Array<T,1>& dst, 
      int n_coef_kept = 0, const bool right_first = false)
    {
      // Checks that the src array has zero base indices
      tca::assertZeroBase( src);

      // Define constant
      const int max_n_coef = src.extent(0)*src.extent(1);

      // if the number of coefficients to be kept is not specified, 
      // set it to the MAX 
      if(0 == n_coef_kept) 
        n_coef_kept = max_n_coef;

      // Checks that the dst array has zero base indices and is of
      // the expected size
      tca::assertZeroBase(dst);
      blitz::TinyVector<int,1> shape( n_coef_kept);
      tca::assertSameShape(dst,shape);
      
      // Check that we ask to keep a valid number of coefficients
      if( n_coef_kept > max_n_coef )
        throw ParamOutOfBoundaryError("n_coef_kept", true, 
          n_coef_kept, max_n_coef);
      if( n_coef_kept < 0 )
        throw ParamOutOfBoundaryError("n_coef_kept", false, 
          n_coef_kept, 0);

      // Apply the zigzag function
      detail::zigzagNoCheck( src, dst, right_first);
    }
	}

	/**
	 * @}
	 */
}

#endif /* BOB5SPRO_IP_ZIGZAG_H */
