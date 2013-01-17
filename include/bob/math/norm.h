/**
 * @file bob/math/norm.h
 * @date Thu Jan 17 17:46:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file implements the Frobenius norm
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

#ifndef BOB_MATH_NORM_H
#define BOB_MATH_NORM_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libmath_api
 * @{
 *
 */
namespace math {

/**
 * @brief Function which implements the Frobenius of a matrix, that is
 * \f$frobenius(A) = \sqrt(\sum_{i,j} |a_{i,j}|^2)\f$
 */
template<typename T>
double frobenius(const blitz::Array<T,2>& A)
{
  return sqrt(blitz::sum(blitz::pow2(A)));
}

}
/**
 * @}
 */
}

#endif /* BOB_MATH_NORM_H */
