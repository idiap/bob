/**
 * @file bob/math/pinv.h
 * @date Tue Jun 18 18:27:22 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to determine the pseudo-inverse
 * using the SVD method.
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

#ifndef BOB_MATH_PINV_H
#define BOB_MATH_PINV_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

/**
 * @brief Function which computes the pseudo-inverse using the SVD method.
 * @warning The output blitz::array B should have the correct 
 *   size, with zero base index. Checks are performed.
 * @param A The A matrix to decompose (size MxN)
 * @param B The pseudo-inverse of the matrix A (size NxM)
 * @param rcond Cutoff for small singular values. Singular values smaller 
 *   (in modulus) than rcond * largest_singular_value (again, in modulus)
 *   are set to zero.
 */
void pinv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B,
  const double rcond=1e-15); 
/**
 * @brief Function which computes the pseudo-inverse using the SVD method.
 * @warning The output blitz::array B should have the correct 
 *   size, with zero base index. Checks are NOT performed.
 * @param A The A matrix to decompose (size MxN)
 * @param B The pseudo-inverse of the matrix A (size NxM)
 * @param rcond Cutoff for small singular values. Singular values smaller 
 *   (in modulus) than rcond * largest_singular_value (again, in modulus)
 *   are set to zero.
 */
void pinv_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B,
  const double rcond=1e-15); 

/**
 * @}
 */
}}

#endif /* BOB_MATH_PINV_H */

