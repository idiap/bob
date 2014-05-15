/**
 * @date Sat Mar 19 22:14:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to determine the SVD decomposition
 * a 2D blitz array using LAPACK.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_SVD_H
#define BOB_MATH_SVD_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

/**
 * @brief Function which performs a 'full' Singular Value Decomposition
 *   using the divide and conquer routine dgesdd of LAPACK.
 * @warning The output blitz::array U, sigma and Vt should have the correct 
 *   size, with zero base index. Checks are performed.
 * @param A The A matrix to decompose (size MxN)
 * @param U The U matrix of left singular vectors (size MxM)
 * @param sigma The vector of singular values (size min(M,N))
 *    Please note that this is a 1D array rather than a 2D diagonal matrix!
 * @param Vt The V^T matrix of right singular vectors (size NxN)
 * @param safe If enabled, use LAPACK dgesvd instead of dgesdd
 */
void svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, blitz::Array<double,2>& Vt, bool safe=false);
/**
 * @brief Function which performs a 'full' Singular Value Decomposition
 *   using the divide and conquer routine dgesdd of LAPACK.
 * @warning The output blitz::array U, sigma and Vt should have the correct 
 *   size, with zero base index. Checks are NOT performed.
 * @param A The A matrix to decompose (size MxN)
 * @param U The U matrix of left singular vectors (size MxM)
 * @param sigma The vector of singular values (size min(M,N))
 *    Please note that this is a 1D array rather than a 2D diagonal matrix!
 * @param Vt The V^T matrix of right singular vectors (size NxN)
 * @param safe If enabled, use LAPACK dgesvd instead of dgesdd
 */
void svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, blitz::Array<double,2>& Vt, bool safe=false);


/**
 * @brief Function which performs a 'partial' Singular Value Decomposition
 *   using the 'simple' driver routine dgesvd of LAPACK. It only returns 
 *   the first min(M,N) columns of U, and is somehow similar to the 
 *   'economical' SVD variant of matlab (except that it does not return V).
 * @warning The output blitz::array U and sigma should have the correct 
 *   size, with zero base index. Checks are performed.
 * @param A The A matrix to decompose (size MxN)
 * @param U The U matrix of left singular vectors (size M x min(M,N))
 * @param sigma The vector of singular values (size min(M,N))
 *    Please note that this is a 1D array rather than a 2D diagonal matrix!
 * @param safe If enabled, use LAPACK dgesvd instead of dgesdd
 */
void svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, bool safe=false);
/**
 * @brief Function which performs a 'partial' Singular Value Decomposition
 *   using the 'simple' driver routine dgesvd of LAPACK. It only returns 
 *   the first min(M,N) columns of U, and is somehow similar to the 
 *   'economical' SVD variant of matlab (except that it does not return V).
 * @warning The output blitz::array U and sigma should have the correct 
 *   size, with zero base index. Checks are NOT performed.
 * @param A The A matrix to decompose (size MxN)
 * @param U The U matrix of left singular vectors (size M x min(M,N))
 * @param sigma The vector of singular values (size min(M,N))
 *    Please note that this is a 1D array rather than a 2D diagonal matrix!
 * @param safe If enabled, use LAPACK dgesvd instead of dgesdd
 */
void svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, bool safe=false);


/**
 * @brief Function which performs a 'partial' Singular Value Decomposition
 *   using the 'simple' driver routine dgesvd of LAPACK. It only returns 
 *   the singular values.
 * @warning The output blitz::array sigma should have the correct 
 *   size, with zero base index. Checks are performed.
 * @param A The A matrix to decompose (size MxN)
 * @param sigma The vector of singular values (size min(M,N))
 *    Please note that this is a 1D array rather than a 2D diagonal matrix!
 * @param safe If enabled, use LAPACK dgesvd instead of dgesdd
 */
void svd(const blitz::Array<double,2>& A, blitz::Array<double,1>& sigma,
  bool safe=false);
/**
 * @brief Function which performs a 'partial' Singular Value Decomposition
 *   using the 'simple' driver routine dgesvd of LAPACK. It only returns 
 *   the singular values.
 * @warning The output blitz::array sigma should have the correct 
 *   size, with zero base index. Checks are NOT performed.
 * @param A The A matrix to decompose (size MxN)
 * @param sigma The vector of singular values (size min(M,N))
 *    Please note that this is a 1D array rather than a 2D diagonal matrix!
 * @param safe If enabled, use LAPACK dgesvd instead of dgesdd
 */
void svd_(const blitz::Array<double,2>& A, blitz::Array<double,1>& sigma,
  bool safe=false);

/**
 * @}
 */
}}

#endif /* BOB_MATH_SVD_H */
