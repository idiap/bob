/**
 * @file bob/math/linear.h
 * @date Thu Mar 31 14:00:25 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines basic matrix and vector operations using 1D and 2D
 * blitz arrays.
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

#ifndef BOB_MATH_LINEAR_H
#define BOB_MATH_LINEAR_H

#include <blitz/array.h>
#include <bob/core/assert.h>

namespace bob { namespace math {

  /**
   * Performs the matrix multiplication C=A*B
   *
   * @warning No checks are performed on the array sizes and is recommended
   * only in scenarios where you have previously checked conformity and is
   * focused only on speed.
   *
   * @param A The A matrix (left element of the multiplication) (size MxN)
   * @param B The B matrix (right element of the multiplication) (size NxP)
   * @param C The resulting matrix (size MxP)
   */
  template<typename T1, typename T2, typename T3>
    void prod_(const blitz::Array<T1,2>& A, const blitz::Array<T2,2>& B,
        blitz::Array<T3,2>& C) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::thirdIndex k;
      C = blitz::sum(A(i,k) * B(k,j), k);
    }

  /**
   * Performs the matrix multiplication C=A*B
   *
   * The input and output data have their sizes checked and this method will
   * raise an appropriate exception if that is not cased. If you know that the
   * input and output matrices conform, use the prod_() variant.
   *
   * @param A The A matrix (left element of the multiplication) (size MxN)
   * @param B The B matrix (right element of the multiplication) (size NxP)
   * @param C The resulting matrix (size MxP)
   */
  template<typename T1, typename T2, typename T3>
    void prod(const blitz::Array<T1,2>& A, const blitz::Array<T2,2>& B,
        blitz::Array<T3,2>& C) {
      // Check inputs
      bob::core::array::assertZeroBase(A);
      bob::core::array::assertZeroBase(B);
      bob::core::array::assertSameDimensionLength(A.extent(1),B.extent(0));

      // Check output
      bob::core::array::assertZeroBase(C);
      bob::core::array::assertSameDimensionLength(A.extent(0), C.extent(0));
      bob::core::array::assertSameDimensionLength(B.extent(1), C.extent(1));

      prod_(A, B, C);
    }

  /**
   * Performs the matrix-vector multiplication c=A*b
   *
   * @warning No checks are performed on the array sizes and is recommended
   * only in scenarios where you have previously checked conformity and is
   * focused only on speed.
   *
   * @param A The A matrix (left element of the multiplication) (size MxN)
   * @param b The b vector (right element of the multiplication) (size N)
   * @param c The resulting vector (size M)
   */
  template<typename T1, typename T2, typename T3>
    void prod_(const blitz::Array<T1,2>& A, const blitz::Array<T2,1>& b,
        blitz::Array<T3,1>& c) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      c = blitz::sum(A(i,j) * b(j), j);
    }

  /**
   * Performs the matrix-vector multiplication c=A*b
   *
   * The input and output data have their sizes checked and this method will
   * raise an appropriate exception if that is not cased. If you know that the
   * input and output matrices conform, use the prod_() variant.
   *
   * @param A The A matrix (left element of the multiplication) (size MxN)
   * @param b The b vector (right element of the multiplication) (size N)
   * @param c The resulting vector (size M)
   */
  template<typename T1, typename T2, typename T3>
    void prod(const blitz::Array<T1,2>& A, const blitz::Array<T2,1>& b,
        blitz::Array<T3,1>& c) {
      // Check inputs
      bob::core::array::assertZeroBase(A);
      bob::core::array::assertZeroBase(b);
      bob::core::array::assertSameDimensionLength(A.extent(1),b.extent(0));

      // Check output
      bob::core::array::assertZeroBase(c);
      bob::core::array::assertSameDimensionLength(c.extent(0), A.extent(0));

      prod_(A, b, c);
    }

  /**
   * Performs the vector-matrix multiplication c=a*B
   *
   * @warning No checks are performed on the array sizes and is recommended
   * only in scenarios where you have previously checked conformity and is
   * focused only on speed.
   *
   * @param a The a vector (left element of the multiplication) (size M)
   * @param B The B matrix (right element of the multiplication) (size MxN)
   * @param c The resulting vector (size N)
   */
  template<typename T1, typename T2, typename T3>
    void prod_(const blitz::Array<T1,1>& a, const blitz::Array<T2,2>& B,
        blitz::Array<T3,1>& c) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      c = blitz::sum(a(j) * B(j,i), j);
    }

  /**
   * Performs the vector-matrix multiplication c=a*B
   *
   * The input and output data have their sizes checked and this method will
   * raise an appropriate exception if that is not cased. If you know that the
   * input and output matrices conform, use the prod_() variant.
   *
   * @param a The a vector (left element of the multiplication) (size M)
   * @param B The B matrix (right element of the multiplication) (size MxN)
   * @param c The resulting vector (size N)
   */
  template<typename T1, typename T2, typename T3>
    void prod(const blitz::Array<T1,1>& a, const blitz::Array<T2,2>& B,
        blitz::Array<T3,1>& c) {
      // Check inputs
      bob::core::array::assertZeroBase(a);
      bob::core::array::assertZeroBase(B);
      bob::core::array::assertSameDimensionLength(a.extent(0),B.extent(0));

      // Check output
      bob::core::array::assertZeroBase(c);
      bob::core::array::assertSameDimensionLength(c.extent(0), B.extent(1));

      prod_(a, B, c);
    }

  /**
   * Performs the outer product between two vectors generating a matrix.
   *
   * @warning No checks are performed on the array sizes and is recommended
   * only in scenarios where you have previously checked conformity and is
   * focused only on speed.
   *
   * @param a The a vector (left element of the multiplication) (size M)
   * @param b The b matrix (right element of the multiplication) (size M)
   * @param C The resulting matrix (size MxM)
   */
  template<typename T1, typename T2, typename T3>
    void prod_(const blitz::Array<T1,1>& a, const blitz::Array<T2,1>& b,
        blitz::Array<T3,2>& C) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      C = a(i) * b(j);
    }

  /**
   * Performs the outer product between two vectors generating a matrix.
   *
   * The input and output data have their sizes checked and this method will
   * raise an appropriate exception if that is not cased. If you know that the
   * input and output matrices conform, use the prod_() variant.
   *
   * @param a The a vector (left element of the multiplication) (size M)
   * @param b The b matrix (right element of the multiplication) (size M)
   * @param C The resulting matrix (size MxM)
   */
  template<typename T1, typename T2, typename T3>
    void prod(const blitz::Array<T1,1>& a, const blitz::Array<T2,1>& b,
        blitz::Array<T3,2>& C) {
      // Check inputs
      bob::core::array::assertZeroBase(a);
      bob::core::array::assertZeroBase(b);

      // Check output
      bob::core::array::assertZeroBase(C);
      bob::core::array::assertSameDimensionLength(C.extent(0), a.extent(0));
      bob::core::array::assertSameDimensionLength(C.extent(1), b.extent(0));

      prod_(a, b, C);
    }

  /**
   * Function which computes the dot product <a,b> between two 1D blitz
   * array.
   *
   * @warning No checks are performed on the array sizes and is recommended
   * only in scenarios where you have previously checked conformity and is
   * focused only on speed.
   *
   * @param a The a vector (size N)
   * @param b The b vector (size N)
   */
  template<typename T1, typename T2>
    T1 dot_(const blitz::Array<T1,1>& a, const blitz::Array<T2,1>& b) {
      return blitz::sum(a * b);
    }

  /**
   * Function which computes the dot product <a,b> between two 1D blitz
   * array.
   *
   * The input data have their sizes checked and this method will raise an
   * appropriate exception if that is not cased. If you know that the input
   * vectors conform, use the dot_() variant.
   *
   * @param a The a vector (size N)
   * @param b The b vector (size N)
   */
  template<typename T1, typename T2>
    T1 dot(const blitz::Array<T1,1>& a, const blitz::Array<T2,1>& b) {
      // Check inputs
      bob::core::array::assertZeroBase(a);
      bob::core::array::assertZeroBase(b);
      bob::core::array::assertSameDimensionLength(a.extent(0),b.extent(0));

      return dot_(a, b);
    }

  /**
   * Computes the trace of a square matrix (the sum of all elements in the main
   * diagonal).
   *
   * @warning No checks are performed on the array extent sizes and is
   * recommended only in scenarios where you have previously checked conformity
   * and is focused only on speed.
   *
   * @param A The input square matrix (size NxN)
   */
  template<typename T> T trace_(const blitz::Array<T,2>& A) {
    blitz::firstIndex i;
    return blitz::sum(A(i,i));
  }

  /**
   * Computes the trace of a square matrix (the sum of all elements in the main
   * diagonal).
   *
   * The input matrix is checked for "square-ness" and raises an appropriate
   * exception if that is not cased. If you know that the input matrix
   * conforms, use the trace_() variant.
   *
   * @param A The input square matrix (size NxN)
   */
  template<typename T> T trace(const blitz::Array<T,2>& A) {
    // Check input
    bob::core::array::assertZeroBase(A);
    bob::core::array::assertSameDimensionLength(A.extent(0),A.extent(1));

    return trace_(A);
  }

  /**
   * Computes the euclidean norm of a vector
   */
  template<typename T> inline double norm(const blitz::Array<T,1>& array) {
    return std::sqrt(blitz::sum(blitz::pow2(array)));
  }

  /**
   * Normalizes a vector 'i' and outputs the normalized vector in 'o'.
   *
   * @warning This version of the normalize() method does not check for length
   * consistencies and is given as an API for cases in which you have done
   * already the check and is focused on speed.
   */
  template<typename T1, typename T2> void normalize_
    (const blitz::Array<T1,1>& i, blitz::Array<T2,1>& o) {
    o = i / norm(i);
  }

  /**
   * Normalizes a vector 'i' and outputs the normalized vector in itself.
   *
   * @note This method receives an array by *value* and not by reference as in
   * many cases we iterate over the vectors in a matrix and we cannot get a
   * non-const reference to a blitz::Array<> slice.
   */
  template<typename T> void normalizeSelf (blitz::Array<T,1> i) {
    i /= norm(i);
  }

  /**
   * Normalizes a vector 'i' and outputs the normalized vector in 'o'.
   *
   * Both vectors are checked to make sure they have the same length. If you
   * want to use an unchecked version, please use normalize_.
   */
  template<typename T1, typename T2> void normalize(const blitz::Array<T1,1>& i,
      blitz::Array<T2,1>& o) {
    // Check input
    bob::core::array::assertSameDimensionLength(i.extent(0),o.extent(0));
    normalize_(i, o);
  }

  /**
   * Generates an eye 2D matrix. If the matrix is squared, then it returns the 
   * identity matrix.
   *
   * @warning No checks are performed on the array and is recommended
   * only in scenarios where you have previously checked conformity and is
   * focused only on speed.
   *
   * @param A The 2D destination matrix (size MxN)
   */
  template<typename T>
    void eye_(blitz::Array<T,2>& A) {
      A = 0.;
      for(int i=0; i<std::min(A.extent(0), A.extent(1)); ++i)
        A(i,i) = 1.;
    }

  /**
   * Generates an eye 2D matrix. If the matrix is squared, then it returns the 
   * identity matrix.
   *
   * @param A The 2D destination matrix (size MxN)
   */
  template<typename T>
    void eye(blitz::Array<T,2>& A) {
      bob::core::array::assertZeroBase(A);
      eye_(A);
    }

  /**
   * Generates a 2D square diagonal matrix from a 1D vector.
   *
   * @warning No checks are performed on the array sizes and is recommended
   * only in scenarios where you have previously checked conformity and is
   * focused only on speed.
   *
   * @param d The 1D vector which contains the diagonal (size N)
   * @param A The 2D destination matrix (size NxN)
   */
  template<typename T>
    void diag_(const blitz::Array<T,1>& d, blitz::Array<T,2>& A) {
      A = 0.;
      for(int i=0; i<A.extent(0); ++i)
        A(i,i) = d(i);
    }

  /**
   * Generates a 2D square diagonal matrix from a 1D vector.
   *
   * @param d The 1D vector which contains the diagonal (size N)
   * @param A The 2D destination matrix (size MxN)
   */
  template<typename T>
    void diag(const blitz::Array<T,1>& d, blitz::Array<T,2>& A) {
      bob::core::array::assertZeroBase(d);
      bob::core::array::assertZeroBase(A);
      bob::core::array::assertSameDimensionLength(d.extent(0),A.extent(0));
      bob::core::array::assertSameDimensionLength(A.extent(0),A.extent(1));
      diag_(d, A);
    }

}}

#endif /* BOB_MATH_LINEAR_H */
