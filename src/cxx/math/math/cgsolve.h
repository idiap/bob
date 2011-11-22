/**
 * @file cxx/math/math/cgsolve.h
 * @date Mon Jun 27 21:14:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions to solve a symmetric positive-definite
 *   linear system A*x=b via conjugate gradients.
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

#ifndef TORCH5SPRO_MATH_CGSOLVE_H
#define TORCH5SPRO_MATH_CGSOLVE_H

#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

    /**
      * @brief Function which solves a symmetric positive-definite linear 
      *   system of equation via conjugate gradients.
      * @param A The A symmetric positive-definite squared-matrix of the 
      *   system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      * @param acc The desired accuracy. The algorithm terminates when
      *   norm(Ax-b)/norm(b) < acc
      * @param max_iter The maximum number of iterations
      */
    void cgsolveSympos(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b, const double acc, const int max_iter);

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_CGSOLVE_H */

