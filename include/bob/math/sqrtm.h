/**
 * @date Wed Oct 12 18:02:26 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to compute the (unique) square root of
 * a real symmetric definite-positive matrix.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_SQRTM_H
#define BOB_MATH_SQRTM_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

    /**
      * @brief Function which computes the (unique) square root of a real 
      *   symmetric definite-positive matrix.
      * @warning The input matrix should be symmetric.
      * @param A The A matrix to decompose (size NxN)
      * @param B The square root matrix B of A (size NxN)
      */
    void sqrtSymReal(const blitz::Array<double,2>& A, blitz::Array<double,2>& B); 
    void sqrtSymReal_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);

/**
 * @}
 */
}}

#endif /* BOB_MATH_SQRTM_H */
