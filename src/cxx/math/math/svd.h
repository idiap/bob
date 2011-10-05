/**
 * @file src/cxx/math/math/svd.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to determine the SVD decomposition 
 * a 2D blitz array using LAPACK.
 * 
 */

#ifndef TORCH5SPRO_MATH_SVD_H
#define TORCH5SPRO_MATH_SVD_H

#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

    /**
      * @brief Function which performs a 'full' Singular Value Decomposition
      *   using the 'generic' dgesvd LAPACK function.
      * @warning The output blitz::array U, sigma and V should have the correct 
      *   size, with zero base index. Checks are performed.
      * @param A The A matrix to decompose (size MxN)
      * @param U The U matrix of left singular vectors (size MxM)
      * @param sigma The vector of singular values (size min(M,N))
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      * @param V The V matrix of right singular vectors (size NxN)
      */
    void svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma, blitz::Array<double,2>& V);
    /**
      * @brief Function which performs a 'full' Singular Value Decomposition
      *   using the 'generic' dgesvd LAPACK function.
      * @warning The output blitz::array U, sigma and V should have the correct 
      *   size, with zero base index. Checks are NOT performed.
      * @param A The A matrix to decompose (size MxN)
      * @param U The U matrix of left singular vectors (size MxM)
      * @param sigma The vector of singular values (size min(M,N))
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      * @param V The V matrix of right singular vectors (size NxN)
      */
    void svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma, blitz::Array<double,2>& V);


    /**
      * @brief Function which performs a 'partial' Singular Value Decomposition
      *   using the 'generic' dgesvd LAPACK function. It only returns the first
      *   min(M,N) columns of U, and is somehow similar to the 'economical' SVD
      *   variant of matlab (except that it does not return V).
      * @warning The output blitz::array U and sigma should have the correct 
      *   size, with zero base index. Checks are performed.
      * @param A The A matrix to decompose (size MxN)
      * @param U The U matrix of left singular vectors (size M x min(M,N))
      * @param sigma The vector of singular values (size min(M,N))
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      */
    void svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma);
    /**
      * @brief Function which performs a 'partial' Singular Value Decomposition
      *   using the 'generic' dgesvd LAPACK function. It only returns the first
      *   min(M,N) columns of U, and is somehow similar to the 'economical' SVD
      *   variant of matlab (except that it does not return V).
      * @warning The output blitz::array U and sigma should have the correct 
      *   size, with zero base index. Checks are NOT performed.
      * @param A The A matrix to decompose (size MxN)
      * @param U The U matrix of left singular vectors (size M x min(M,N))
      * @param sigma The vector of singular values (size min(M,N))
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      */
    void svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma);
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_SVD_H */

