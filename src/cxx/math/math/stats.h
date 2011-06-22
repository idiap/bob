/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 17 Jun 20:44:54 2011 
 *
 * @brief Some statistical base methods
 */

#ifndef TORCH_MATH_STATS_H 
#define TORCH_MATH_STATS_H

#include <blitz/array.h>
#include "core/array_assert.h"

namespace Torch { namespace math {

    /**
     * Computes the scatter matrix of a 2D array considering data is
     * organized column-wise (each sample is a column, each feature is a row).
     * Outputs the sample mean M and the scatter matrix S.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     *
     * This version of the method also returns the sample mean of the array.
     * The resulting arrays (M and S) will be resized if required.
     */
    template<typename T>
    void scatter_(const blitz::Array<T,2>& A, blitz::Array<T,2>& S, 
        blitz::Array<T,1>& M) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::Range a = blitz::Range::all();

      M = blitz::mean(A,j);
      S = 0;

      blitz::Array<T,1> buffer(A.extent(0));
      for (int z=0; z<A.extent(1); ++z) {
        buffer = A(a,z) - M;
        S += buffer(i) * buffer(j); //outer product
      }
    }

    /**
     * Computes the scatter matrix of a 2D array considering data is
     * organized column-wise (each sample is a column, each feature is a row).
     * Outputs the sample mean M and the scatter matrix S.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that
     * the input and output matrices conform, use the scatter_() variant.
     *
     * This version of the method also returns the sample mean of the array.
     * The resulting arrays (M and S) will be resized if required.
     */
    template<typename T>
    void scatter(const blitz::Array<T,2>& A, blitz::Array<T,2>& S, 
        blitz::Array<T,1>& M) {

      // Check output
      Torch::core::array::assertSameDimensionLength(A.extent(0), M.extent(0));
      Torch::core::array::assertSameDimensionLength(A.extent(0), S.extent(0));
      Torch::core::array::assertSameDimensionLength(A.extent(0), S.extent(1));

      scatter_<T>(A, S, M);
    }

    /**
     * Computes the scatter matrix of a 2D array considering data is
     * organized column-wise (each sample is a column, each feature is a row).
     * Outputs the sample scatter matrix S.
     *
     * @warning No checks are performed on the array sizes and is recommended
     * only in scenarios where you have previously checked conformity and is
     * focused only on speed.
     *
     * The input array S is resized if necessary.
     */
    template<typename T>
    void scatter_(const blitz::Array<T,2>& A, blitz::Array<T,2>& S) {
      blitz::Array<T,1> M;
      scatter_<T>(A, S, M);
    }

    /**
     * Computes the scatter matrix of a 2D array considering data is
     * organized column-wise (each sample is a column, each feature is a row).
     * Outputs the sample scatter matrix S.
     *
     * The input and output data have their sizes checked and this method will
     * raise an appropriate exception if that is not cased. If you know that
     * the input and output matrices conform, use the scatter_() variant.
     *
     * The input array S is resized if necessary.
     */
    template<typename T>
    void scatter(const blitz::Array<T,2>& A, blitz::Array<T,2>& S) {
      blitz::Array<T,1> M;
      scatter<T>(A, S, M);
    }

}}

#endif /* TORCH_MATH_STATS_H */
