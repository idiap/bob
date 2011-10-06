/**
 * @file src/cxx/math/math/cgsolve.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines functions to solve a symmetric positive-definite 
 *   linear system A*x=b via conjugate gradients.
 * 
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
    void cgsolveSympos_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b, const double acc, const int max_iter);

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_CGSOLVE_H */

