/**
 * @file src/cxx/math/math/linprog.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to solve a linear system using LAPACK.
 * 
 */

#ifndef TORCH5SPRO_MATH_LINPROG_H
#define TORCH5SPRO_MATH_LINPROG_H 1

#include "core/logging.h"
#include "core/Exception.h"
//#include "math/Exception.h"

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
      * @warning The X blitz::array is resized and reindexed with zero 
      *   base index.
      * @param A The A squared-matrix of the system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      */
    void linsolve(const blitz::Array<double,2>& A, blitz::Array<double,1>& x, 
      const blitz::Array<double,1>& b);
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_LINPROG_H */

