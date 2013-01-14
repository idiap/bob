/**
 * @file bob/math/svd.h
 * @date Sat Mar 19 22:14:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to determine the SVD decomposition
 * a 2D blitz array using LAPACK.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_MATH_SVD_H
#define BOB_MATH_SVD_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

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
      */
    void svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma, blitz::Array<double,2>& Vt);
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
      */
    void svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma, blitz::Array<double,2>& Vt);


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
      */
    void svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma);
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
      */
    void svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, 
      blitz::Array<double,1>& sigma);


    /**
      * @brief Function which performs a 'partial' Singular Value Decomposition
      *   using the 'simple' driver routine dgesvd of LAPACK. It only returns 
      *   the singular values.
      * @warning The output blitz::array sigma should have the correct 
      *   size, with zero base index. Checks are performed.
      * @param A The A matrix to decompose (size MxN)
      * @param sigma The vector of singular values (size min(M,N))
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      */
    void svd(const blitz::Array<double,2>& A, blitz::Array<double,1>& sigma);
    /**
      * @brief Function which performs a 'partial' Singular Value Decomposition
      *   using the 'simple' driver routine dgesvd of LAPACK. It only returns 
      *   the singular values.
      * @warning The output blitz::array sigma should have the correct 
      *   size, with zero base index. Checks are NOT performed.
      * @param A The A matrix to decompose (size MxN)
      * @param sigma The vector of singular values (size min(M,N))
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      */
    void svd_(const blitz::Array<double,2>& A, blitz::Array<double,1>& sigma);

  }
/**
 * @}
 */
}

#endif /* BOB_MATH_SVD_H */

