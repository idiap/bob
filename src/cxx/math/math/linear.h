/**
 * @file src/cxx/math/math/linear.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines basic matrix and vector operation using 1D and 2D
 * blitz arrays.
 * 
 */

#ifndef TORCH5SPRO_MATH_LINEAR_H
#define TORCH5SPRO_MATH_LINEAR_H 1

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
      // Check that the dimensions of the input arrays are compatible for
      // matrix multiplication
      if( A.extent(1) != B.extent(0) || A.base(1) != B.base(0) )
        throw Torch::core::Exception();

      // Reindex and resize output array
      blitz::reindexAndResize( C, A.base(0), B.base(1), A.extent(0), 
        B.extent(1));

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
      // Check that the dimensions of the input arrays are compatible for
      // matrix multiplication
      if( A.extent(1) != b.extent(0) || A.base(1) != b.base(0) )
        throw Torch::core::Exception();

      // Reindex and resize output array
      blitz::reindexAndResize( c, A.base(0), A.extent(0));

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
      // Check that the dimensions of the input arrays are compatible for
      // the dot product
      if( a.extent(0) != b.extent(0) || a.base(0) != b.base(0) )
        throw Torch::core::Exception();

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
