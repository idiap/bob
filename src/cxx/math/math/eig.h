/**
 * @file src/cxx/math/math/eig.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to determine an eigenvalue decomposition 
 * a 2D blitz array using LAPACK.
 * 
 */

#ifndef TORCH5SPRO_MATH_EIG_H
#define TORCH5SPRO_MATH_EIG_H

#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

    /**
      * @brief Function which performs an eigenvalue decomposition of a real
      *   symmetric matrix, using the dsyev LAPACK function.
      * @warning The input matrix should be symmetric.
      * @param A The A matrix to decompose (size NxN)
      * @param V The V matrix of eigenvectors (size NxN) stored in columns
      * @param D The vector of eigenvalues (size N)
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      */
    void eigSymReal(const blitz::Array<double,2>& A, blitz::Array<double,2>& V, 
      blitz::Array<double,1>& D);

    /**
      * @brief Computes all the eigenvalues and the eigenvectors of a real 
      *   generalized symmetric-definite eigenproblem, of the form:
      *   A*x=(lambda)*B*x, using the dsygv LAPACK function.
      * @warning The input matrices A and B are assumed to be symmetric and B
      *   is also positive definite.
      * @param A The A input matrix (size NxN) of the problem
      * @param B The B input matrix (size NxN) of the problem
      * @param V The V matrix of eigenvectors (size NxN) stored in columns
      * @param D The vector of eigenvalues (size N)
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      */
    void eigSym(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B,
      blitz::Array<double,2>& V, blitz::Array<double,1>& D);

    /**
      * @brief Computes all the eigenvalues and the eigenvectors of a real 
      *   generalized eigenproblem, of the form:
      *   A*x=(lambda)*B*x, using the dggev LAPACK function.
      * @param A The A input matrix (size NxN) of the problem
      * @param B The B input matrix (size NxN) of the problem
      * @param V The V matrix of eigenvectors (size NxN) stored in columns
      * @param D The vector of eigenvalues (size N)
      *    Please note that this is a 1D array rather than a 2D diagonal matrix!
      *
      * AA: Note the returned vectors and values are pre-sorted in decreasing
      * eigen-value order.
      */
    void eig(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B,
      blitz::Array<double,2>& V, blitz::Array<double,1>& D);
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_EIG_H */

