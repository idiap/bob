/**
 * @file cxx/math/math/inv.h
 * @date Tue Jun 7 01:00:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines function to inverse a 2D blitz array matrix.
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

#ifndef BOB_MATH_INV_H
#define BOB_MATH_INV_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {
    /**
      * @brief Function which computes the inverse of a matrix,
      *   using the dgetrf and dgetri LAPACK functions.
      * @param A The A matrix to decompose (size NxN)
      * @param B The B=inverse(A) matrix (size NxN)
      */
    void inv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);
    void inv_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);
  }
/**
 * @}
 */
}

#endif /* BOB_MATH_INV_H */
