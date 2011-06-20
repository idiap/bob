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
     */
    template<typename T>
    void scatter(const blitz::Array<T,2>& A, blitz::Array<T,1>& S) {
      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::Range a = blitz::Range::all();
      blitz::Array<T,1> means(A.extent(0));
      means = blitz::mean(A,j);
      blitz::Array<T,1> buffer(A.extent(0));
      for (blitz::sizeType z=0; z<A.extent(0); ++z) {
        buffer = A(a,z) - means;
        S += buffer(i) * buffer(j); //outer product
      }
    }

}}

#endif /* TORCH_MATH_STATS_H */

