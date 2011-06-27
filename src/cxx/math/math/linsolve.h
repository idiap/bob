/**
 * @file src/cxx/math/math/linprog.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines functions to solve linear systems using LAPACK.
 * 
 */

#ifndef TORCH5SPRO_MATH_LINPROG_H
#define TORCH5SPRO_MATH_LINPROG_H

#include <blitz/array.h>

namespace Torch {
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
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_LINPROG_H */

