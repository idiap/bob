/**
 * @file src/cxx/math/math/linprog.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file define interior point methods  which allow to solve a 
 *        linear program (LP).
 * 
 */

#ifndef TORCH5SPRO_MATH_INTERIOR_POINT_H
#define TORCH5SPRO_MATH_INTERIOR_POINT_H 1

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

    bool isFeasible(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda, 
      const blitz::Array<double,1>& mu, const double epsilon);

    bool isInV2(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
      const blitz::Array<double,1>& mu, const double epsilon, const double theta);

    bool isInVinf(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
      const blitz::Array<double,1>& mu, const double epsilon, const double gamma);

    /**
      * @brief Function which solves a linear program using a short-step
      *   interior point method.
      * @warning The X blitz::array is resized and reindexed with zero 
      *   base index.
      * @param A The A squared-matrix of the system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      */
    void interiorpointShortstep(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double theta, blitz::Array<double,1>& x, 
      blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
      const double epsilon);

    void interiorpointPredictorCorrector(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double theta_pred, const double theta_corr, blitz::Array<double,1>& x, 
      blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
      const double epsilon);

    void interiorpointLongstep(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double gamma, const double sigma, blitz::Array<double,1>& x, 
      blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
      const double epsilon);
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_LINPROG_H */

