/**
 * @file cxx/math/math/sqrtm.h
 * @date Wed Oct 12 18:02:26 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to compute the (unique) square root of
 * a real symmetric definite-positive matrix.
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

#ifndef BOB_MATH_SQRTM_H
#define BOB_MATH_SQRTM_H

#include <blitz/array.h>
#include "eig.h"

namespace bob {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

    /**
      * @brief Function which computes the (unique) square root of a real 
      *   symmetric definite-positive matrix.
      * @warning The input matrix should be symmetric.
      * @param A The A matrix to decompose (size NxN)
      * @param B The square root matrix B of A (size NxN)
      */
    void sqrtSymReal(const blitz::Array<double,2>& A, blitz::Array<double,2>& B); 
    void sqrtSymReal_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);

  }
/**
 * @}
 */
}

#endif /* BOB_MATH_SQRTM_H */
