/**
 * @file cxx/io/io/reorder.h
 * @date Tue Nov 22 11:24:44 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Row-major to column-major reordering and vice-versa
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

#ifndef BOB_IO_REORDER_H 
#define BOB_IO_REORDER_H

#include "core/array.h"
#include <stdint.h>

namespace bob { namespace io {

  /**
   * Returns, on the first argument, the linear indexes by calculating the
   * linear positions relative to both row-major and column-major order
   * matrixes given a certain index accessing a position in the matrix and the
   * matrix shape
   *
   * @param linear indexes (row, col): a 2-tuple with the results: row-major
   *        and column-major linear indexes
   * @param indexes (i,j) a 2-tuple with the indexes as would be accessed
   *        [col][row]; this is the same as accessing the matrix like on
   *        directions [y][x]
   * @param shape a 2-tuple with the matrix shape like [col][row]; this is the
   *        same as thinking about the extends of the matrix like on directions
   *        [y][x]
   *
   * Detailed arithmetics with graphics and explanations can be found here:
   * http://webster.cs.ucr.edu/AoA/Windows/HTML/Arraysa2.html
   */
  void rc2d(size_t& row, size_t& col, const size_t i, const size_t j,
      const size_t* shape);

  /**
   * Same as above, but for a 3D array organized as [depth][column][row]
   */
  void rc3d(size_t& row, size_t& col, const size_t i, const size_t j,
      const size_t k, const size_t* shape);

  /**
   * Same as above, but for a 4D array organized as [time][depth][column][row]
   */
  void rc4d(size_t& row, size_t& col, const size_t i, const size_t j,
      const size_t k, const size_t l, const size_t* shape);

  /**
   * Converts the data from row-major order (C-Style) to column major order
   * (Fortran style), which is required by matio. Input parameters are the src
   * data in row-major order, the destination (pre-allocated) array of the same
   * size and the type information.
   */
  void row_to_col_order(const void* src_, void* dst_, 
      const bob::core::array::typeinfo& info);

  /**
   * Converts the data from column-major order (Fortran-Style) to row major
   * order (C style), which is required by bob. Input parameters are the src
   * data in column-major order, the destination (pre-allocated) array of the
   * same size and the type information.
   */
  void col_to_row_order(const void* src_, void* dst_, 
      const bob::core::array::typeinfo& info);

  /**
   * Converts the data from row-major order (C-Style) to column major order
   * (Fortran style), which is required by matio. Input parameters are the src
   * data in row-major order, the destination (pre-allocated) array of the same
   * size and the type information.
   */
  void row_to_col_order_complex(const void* src_, void* dst_re_,
      void* dst_im_, const bob::core::array::typeinfo& info);

  /**
   * Converts the data from column-major order (Fortran-Style) to row major
   * order (C style), which is required by bob. Input parameters are the src
   * data in column-major order, the destination (pre-allocated) array of the
   * same size and the type information.
   */
  void col_to_row_order_complex(const void* src_re_, const void* src_im_,
      void* dst_, const bob::core::array::typeinfo& info);
  
}}

#endif /* BOB_IO_REORDER_H */
