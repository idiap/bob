/**
 * @file cxx/math/src/histogram.h
 * @date Mon Apr 16
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Implements fast versions of some histogram measures
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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


#ifndef BOB_MATH_HISTOGRAM_H
#define BOB_MATH_HISTOGRAM_H

#include <core/array.h>
#include <core/array_assert.h>
#include <blitz/array.h>
#include <numeric>
#include <functional>

namespace bob{
  namespace math{

    template <class T>
      // wrapper for the std::min function (required to be compilable under OsX)
      static inline T minimum(const T& v1, const T& v2){
        return std::min<T>(v1,v2);
      }

    template <class T>
      // helper function to compute the chi_square distance between the given values
      static inline T chi_square_distance(const T& v1, const T& v2){
        return v1 != v2 ? (v1 - v2) * (v1 - v2) / (v1 + v2) : T(0);
      }


    template <class T>
      //! Fast implementation of the histogram intersection measure
      inline T histogram_intersection(const blitz::Array<T,1>& h1, const blitz::Array<T,1>& h2){
        bob::core::array::assertCContiguous(h1);
        bob::core::array::assertCContiguous(h2);
        bob::core::array::assertSameShape(h1,h2);
        // we use the std::inner_product function (using blitz iterators!),
        // but instead of computing the element-wise multiplication,
        // we use the std::min of the two elements
        return std::inner_product(
          h1.begin(),
          h1.end(),
          h2.begin(),
          T(0),
          std::plus<T>(),
          bob::math::minimum<T>
        );
      }

    template <class T>
      //! Fast implementation of the chi square histogram distance measure
      inline T chi_square(const blitz::Array<T,1>& h1, const blitz::Array<T,1>& h2){
        bob::core::array::assertCContiguous(h1);
        bob::core::array::assertCContiguous(h2);
        bob::core::array::assertSameShape(h1,h2);
        // we use the std::inner_product function (using blitz iterators!),
        // but instead of computing the element-wise multiplication,
        // we use our own chi_square_distance function
        return std::inner_product(
          h1.begin(),
          h1.end(),
          h2.begin(),
          T(0),
          std::plus<T>(),
          &bob::math::chi_square_distance<T>
        );
      }
  } // namespace math
} // namespace bob

#endif // BOB_MATH_HISTOGRAM_H
