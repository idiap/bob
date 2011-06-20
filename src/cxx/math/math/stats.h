/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 17 Jun 20:44:54 2011 
 *
 * @brief Some statistical base methods
 */

#ifndef TORCH_MATH_STATS_H 
#define TORCH_MATH_STATS_H

#include <blitz/array.h>
#include "core/blitz_compat.h"

namespace Torch { namespace math {

    /**
     * Computes the scatter matrix of a 2D array considering data is
     * organized column-wise (each sample is a column, each feature is a row).
     *
     * This version of the method also returns the sample mean of the array.
     * The resulting arrays (M and S) will be resized if required.
     */
    template<typename T>
    void scatter(const blitz::Array<T,2>& A, blitz::Array<T,1>& M, 
        blitz::Array<T,2>& S) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::Range a = blitz::Range::all();

      M.resize(A.extent(0));
      M = blitz::mean(A,j);
      S.resize(A.extent(0), A.extent(0));
      S = 0;

      blitz::Array<T,1> buffer(A.extent(0));
      for (int z=0; z<A.extent(0); ++z) {
        buffer = A(a,z) - M;
        S += buffer(i) * buffer(j); //outer product
      }
    }

    /**
     * Computes the scatter matrix of a 2D array considering data is
     * organized column-wise (each sample is a column, each feature is a row).
     *
     * The input array S is resized if necessary.
     */
    template<typename T>
    void scatter(const blitz::Array<T,2>& A, blitz::Array<T,2>& S) {
      blitz::Array<T,1> M;
      scatter<T>(A, M, S);
    }

}}

#endif /* TORCH_MATH_STATS_H */

