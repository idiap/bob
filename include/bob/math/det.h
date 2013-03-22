/**
 * @file bob/math/det.h
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines function to compute the determinant of
 *   a 2D blitz array matrix.
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

#ifndef BOB_MATH_DET_H
#define BOB_MATH_DET_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

/**
 * @brief Function which computes the determinant of a square matrix
 * @param A The A matrix to consider (size NxN)
 */
double det(const blitz::Array<double,2>& A);
/**
 * @brief Function which computes the determinant of a square matrix
 * @param A The A matrix to consider (size NxN)
 * @warning Does not check the input matrix
 */
double det_(const blitz::Array<double,2>& A);

/**
 * @}
 */
}}

#endif /* BOB_MATH_DET_H */
