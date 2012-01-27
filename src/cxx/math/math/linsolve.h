/**
 * @file cxx/math/math/linsolve.h
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions to solve linear systems
 *   A*x=b using LAPACK or a conjugate gradient implementation.
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

#ifndef BOB_MATH_LINSOLVE_H
#define BOB_MATH_LINSOLVE_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

    /**
      * @brief Function which solves a linear system of equation using the
      *   'generic' dgsev LAPACK function.
      * @param A The A squared-matrix of the system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      */
    void linsolve(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b);
    void linsolve_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b);

    /**
      * @brief Function which solves a symmetric positive definite linear 
      *   system of equation using the dposv LAPACK function.
      * @warning No check is performed wrt. to the fact that A should be
      *   symmetric positive definite.
      * @param A The A squared-matrix, symmetric definite positive, of the 
      *   system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      */
    void linsolveSympos(const blitz::Array<double,2>& A, 
      blitz::Array<double,1>& x, const blitz::Array<double,1>& b); 
    void linsolveSympos_(const blitz::Array<double,2>& A, 
      blitz::Array<double,1>& x, const blitz::Array<double,1>& b); 

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
    void linsolveCGSympos(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b, const double acc, const int max_iter);
    void linsolveCGSympos_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b, const double acc, const int max_iter);

  }
/**
 * @}
 */
}

#endif /* BOB_MATH_LINSOLVE_H */

