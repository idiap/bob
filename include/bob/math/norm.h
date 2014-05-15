/**
 * @date Thu Jan 17 17:46:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file implements the Frobenius norm
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_NORM_H
#define BOB_MATH_NORM_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

/**
 * @brief Function which implements the Frobenius of a matrix, that is
 * \f$frobenius(A) = \sqrt(\sum_{i,j} |a_{i,j}|^2)\f$
 */
template<typename T>
double frobenius(const blitz::Array<T,2>& A)
{
  return sqrt(blitz::sum(blitz::pow2(A)));
}

/**
 * @}
 */
}}

#endif /* BOB_MATH_NORM_H */
