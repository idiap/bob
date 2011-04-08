/**
 * @file src/cxx/math/math/linear.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines basic matrix and vector operations using 1D and 2D
 * blitz arrays.
 * 
 */

#ifndef TORCH5SPRO_MATH_LINEAR_H
#define TORCH5SPRO_MATH_LINEAR_H

#include "core/logging.h"
#include "core/common.h"

namespace Torch {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

    /**
      * @brief Function which performs the matrix multiplication C=A*B
      * @warning The output blitz::array U, sigma and S are resized and 
      *   reindexed with zero base index.
      * @param A The A matrix (left element of the multiplication) (size MxN)
      * @param B The B matrix (right element of the multiplication) (size NxP)
      * @param C The resulting matrix (size MxP)
      */
    template<typename T>
    void prod(const blitz::Array<T,2>& A, const blitz::Array<T,2>& B,
      blitz::Array<T,2>& C)
    {
      // Check inputs
      Torch::core::assertZeroBase(A);
      Torch::core::assertZeroBase(B);
      Torch::core::assertSameDimensionLength(A.extent(1),B.extent(0));

      // Check output
      Torch::core::assertZeroBase(C);
      Torch::core::assertSameDimensionLength(A.extent(0), C.extent(0));
      Torch::core::assertSameDimensionLength(B.extent(1), C.extent(1));

      // Perform multiplication
      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::thirdIndex k;
      C = sum(A(i,k) * B(k,j), k);
    }

    template <typename T>
    void prod(const blitz::Array<T,2>& A, const blitz::Array<T,1>& b,
      blitz::Array<T,1>& c)
    {
      // Check inputs
      Torch::core::assertZeroBase(A);
      Torch::core::assertZeroBase(b);
      Torch::core::assertSameDimensionLength(A.extent(1),b.extent(0));

      // Check output
      Torch::core::assertZeroBase(c);
      Torch::core::assertSameDimensionLength(c.extent(0), A.extent(0));

      // Perform multiplication
      blitz::firstIndex i;
      blitz::secondIndex j;
      c = sum(A(i,j) * b(j), j);
    }

    /**
      * @brief Function which computes the dot product <a,b> between two 1D
      * blitz array.
      * @param a The a vector (size N)
      * @param b The b vector (size N)
      */
    template<typename T>
    T dot(const blitz::Array<T,1>& a, const blitz::Array<T,1>& b)
    {
      // Check inputs
      Torch::core::assertZeroBase(a);
      Torch::core::assertZeroBase(b);
      Torch::core::assertSameDimensionLength(a.extent(0),b.extent(0));

      // Compute the dot product
      blitz::firstIndex i;
      return sum(a(i) * b(i));
    }
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_LINEAR_H */
