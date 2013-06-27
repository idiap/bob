/**
 * @file bob/math/eig.h
 * @date Mon May 16 21:45:27 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file defines a function to determine an eigenvalue decomposition
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

#ifndef BOB_MATH_EIG_H
#define BOB_MATH_EIG_H

#include <blitz/array.h>
#include <complex>

namespace bob { namespace math {

  /**
   * @ingroup MATH
   * @{
   */

  /**
   * @brief Function which performs an eigenvalue decomposition of a real
   * matrix, using the LAPACK function <code>dgeev</code>.
   *
   * @param A The A matrix to decompose (size NxN)
   *
   * @param V The V matrix of eigenvectors (size NxN) stored in columns
   *
   * @param D The vector of eigenvalues (size N) (in ascending order). Note
   * that this is a 1D array rather than a 2D diagonal matrix!
   */
  void eig(const blitz::Array<double,2>& A, 
      blitz::Array<std::complex<double>,2>& V, 
      blitz::Array<std::complex<double>,1>& D);
  
  /**
   * @brief Function which performs an eigenvalue decomposition of a real
   * matrix, using the LAPACK function <code>dgeev</code>. This version does
   * <b>NOT</b> perform any checks on the input data and should be, therefore,
   * faster.
   *
   * @param A The A matrix to decompose (size NxN)
   *
   * @param V The V matrix of eigenvectors (size NxN) stored in columns
   *
   * @param D The vector of eigenvalues (size N) (in ascending order). Note
   * that this is a 1D array rather than a 2D diagonal matrix!
   */
  void eig_(const blitz::Array<double,2>& A,
      blitz::Array<std::complex<double>,2>& V, 
      blitz::Array<std::complex<double>,1>& D);

  /**
   * @brief Function which performs an eigenvalue decomposition of a real
   * symmetric matrix, using the LAPACK function <code>dsyevd</code>.
   *
   * @warning The input matrix should be symmetric.
   *
   * @param A The A matrix to decompose (size NxN)
   *
   * @param V The V matrix of eigenvectors (size NxN) stored in columns
   *
   * @param D The vector of eigenvalues (size N) (in ascending order). Note
   * that this is a 1D array rather than a 2D diagonal matrix!
   */
  void eigSym(const blitz::Array<double,2>& A, blitz::Array<double,2>& V, 
      blitz::Array<double,1>& D);

  /**
   * @brief Function which performs an eigenvalue decomposition of a real
   * symmetric matrix, using the LAPACK function <code>dsyevd</code>. This
   * version does <b>NOT</b> perform any checks on the input data and should
   * be, therefore, faster.
   *
   * @warning The input matrix should be symmetric.
   *
   * @param A The A matrix to decompose (size NxN)
   *
   * @param V The V matrix of eigenvectors (size NxN) stored in columns
   *
   * @param D The vector of eigenvalues (size N) (in ascending order). Note
   * that this is a 1D array rather than a 2D diagonal matrix!
   */
  void eigSym_(const blitz::Array<double,2>& A, blitz::Array<double,2>& V, 
      blitz::Array<double,1>& D);

  /**
   * @brief Computes all the eigenvalues and the eigenvectors of a real
   * generalized symmetric-definite eigenproblem, of the form:
   * A*x=(lambda)*B*x, using the LAPACK function <code>dsygvd</code>.
   *
   * @warning The input matrices A and B are assumed to be symmetric and B is
   * also positive definite.
   *
   * @param A The A input matrix (size NxN) of the problem
   *
   * @param B The B input matrix (size NxN) of the problem
   *
   * @param V The V matrix of eigenvectors (size NxN) stored in columns
   *
   * @param D The vector of eigenvalues (size N) (in ascending order). Note
   * that this is a 1D array rather than a 2D diagonal matrix!
   */
  void eigSym(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B,
      blitz::Array<double,2>& V, blitz::Array<double,1>& D);

  /**
   * @brief Computes all the eigenvalues and the eigenvectors of a real
   * generalized symmetric-definite eigenproblem, of the form:
   * A*x=(lambda)*B*x, using the LAPACK function <code>dsygvd</code>. This
   * version does <b>NOT</b> perform any checks on the input data and should
   * be, therefore, faster.
   *
   * @warning The input matrices A and B are assumed to be symmetric and B is
   * also positive definite.
   *
   * @param A The A input matrix (size NxN) of the problem
   *
   * @param B The B input matrix (size NxN) of the problem
   *
   * @param V The V matrix of eigenvectors (size NxN) stored in columns
   *
   * @param D The vector of eigenvalues (size N) (in ascending order). Note
   * that this is a 1D array rather than a 2D diagonal matrix!
   */
  void eigSym_(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B,
      blitz::Array<double,2>& V, blitz::Array<double,1>& D);

  /**
   * @}
   */

}}

#endif /* BOB_MATH_EIG_H */
