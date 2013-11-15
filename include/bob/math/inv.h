/**
 * @file bob/math/inv.h
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines function to inverse a 2D blitz array matrix.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_INV_H
#define BOB_MATH_INV_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

/**
 * @brief Function which computes the inverse of a matrix,
 *   using the dgetrf and dgetri LAPACK functions.
 * @param A The A matrix to decompose (size NxN)
 * @param B The B=inverse(A) matrix (size NxN)
 */
void inv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);
void inv_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);

/**
 * @}
 */
}}

#endif /* BOB_MATH_INV_H */
