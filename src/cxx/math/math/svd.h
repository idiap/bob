/**
 * @file src/cxx/math/math/svd.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to determine the SVD decomposition 
 * a 2D blitz array using LAPACK.
 * 
 */

#ifndef TORCH5SPRO_MATH_SVD_H
#define TORCH5SPRO_MATH_SVD_H 1

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
      * @brief Function which performs a Singular Value Decomposition using the
      *   'generic' dgesvd LAPACK function.
      * @warning The output blitz::array U, sigma and S are resized and 
      *   reindexed with zero base index.
      * @param A The A matrix to decompose (size MxN)
      * @param U The U matrix of left singular vectors (size MxM)
      * @param sigma The vector of singular values (size min(M,N))
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      * @param V The V matrix of right singular vectors (size NxN)
      */
    void svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma, blitz::Array<double,2>& V);
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_SVD_H */

