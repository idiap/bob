/**
 * @file bob/math/histogram.h
 * @date Mon Apr 16
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Implements fast versions of some histogram measures
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


#ifndef BOB_MATH_HISTOGRAM_H
#define BOB_MATH_HISTOGRAM_H

#include <bob/core/assert.h>
#include <blitz/array.h>
#include <numeric>
#include <functional>
#include <cmath>

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
      // helper function to compute the symmetric Kullback-Leibler divergence
      static inline double kullback_leibler_divergence(const T& v1, const T& v2){
        // This implementation is inspired by http://www.informatik.uni-freiburg.de/~tipaldi/FLIRTLib/HistogramDistances_8hpp_source.html
        // and http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/WAHL1/node5.html
        double a1 = std::max((double)v1, 1e-5);
        double a2 = std::max((double)v2, 1e-5);
        return (a1 - a2) * std::log(a1/a2);
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

    template <class T1, class T2>
      //! Fast implementation of the sparse histogram intersection measure
      inline T2 histogram_intersection(
            const blitz::Array<T1,1>& index_1, const blitz::Array<T2,1>& values_1,
            const blitz::Array<T1,1>& index_2, const blitz::Array<T2,1>& values_2
      ){
        bob::core::array::assertSameShape(index_1,values_1);
        bob::core::array::assertSameShape(index_2,values_2);
        int i1 = 0, i2 = 0, i1_end = index_1.shape()[0], i2_end = index_2.shape()[0];
        T1 p1 = index_1(i1), p2 = index_2(i2);
        T2 sum = T2(0);
        while (i1 < i1_end && i2 < i2_end){
          p1 = index_1(i1);
          p2 = index_2(i2);
          if (p1 == p2) sum += bob::math::minimum(values_1(i1++), values_2(i2++));
          else if (p1 < p2) ++i1;
          else ++i2;
        }
        return sum;
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
          bob::math::chi_square_distance<T>
        );
      }

    template <class T1, class T2>
      //! Fast implementation of the sparse chi square measure
      inline T2 chi_square(
            const blitz::Array<T1,1>& index_1, const blitz::Array<T2,1>& values_1,
            const blitz::Array<T1,1>& index_2, const blitz::Array<T2,1>& values_2
      ){
        bob::core::array::assertSameShape(index_1,values_1);
        bob::core::array::assertSameShape(index_2,values_2);
        int i1 = 0, i2 = 0, i1_end = index_1.shape()[0], i2_end = index_2.shape()[0];
        T1 p1 = index_1(i1), p2 = index_2(i2);
        T2 sum = T2(0);
        while (i1 < i1_end && i2 < i2_end){
          p1 = index_1(i1);
          p2 = index_2(i2);
          if (p1 == p2){
            sum += bob::math::chi_square_distance(values_1(i1++), values_2(i2++));
          } else if (p1 < p2) {
            sum += bob::math::chi_square_distance(values_1(i1++), T2(0));
          } else{
            sum += bob::math::chi_square_distance(T2(0), values_2(i2++));
          }
        }
        // roll up the remaining parts of the histograms
        while (i1 < i1_end) sum += chi_square_distance(values_1(i1++), T2(0));
        while (i2 < i2_end) sum += chi_square_distance(T2(0), values_2(i2++));
        return sum;
      }

    template <class T>
      //! Fast implementation of the symmetric Kullback-Leibler histogram divergence measure (a distance measure)
      inline double kullback_leibler(const blitz::Array<T,1>& h1, const blitz::Array<T,1>& h2){
        bob::core::array::assertCContiguous(h1);
        bob::core::array::assertCContiguous(h2);
        bob::core::array::assertSameShape(h1,h2);
        // we use the std::inner_product function (using blitz iterators!)
        // calling out implementation of the Kullback-Leibler divergence of two values
        return std::inner_product(
          h1.begin(),
          h1.end(),
          h2.begin(),
          0.,
          std::plus<T>(),
          bob::math::kullback_leibler_divergence<T>
        );
      }

    template <class T1, class T2>
      //! Fast implementation of the sparse symmetric Kullback-Leibler histogram divergence measure (a distance measure)
      inline double kullback_leibler(
            const blitz::Array<T1,1>& index_1, const blitz::Array<T2,1>& values_1,
            const blitz::Array<T1,1>& index_2, const blitz::Array<T2,1>& values_2
      ){
        bob::core::array::assertSameShape(index_1,values_1);
        bob::core::array::assertSameShape(index_2,values_2);
        int i1 = 0, i2 = 0, i1_end = index_1.shape()[0], i2_end = index_2.shape()[0];
        T1 p1 = index_1(i1), p2 = index_2(i2);
        double sum = 0.;
        while (i1 < i1_end && i2 < i2_end){
          p1 = index_1(i1);
          p2 = index_2(i2);
          if (p1 == p2){
            sum += bob::math::kullback_leibler_divergence(values_1(i1++), values_2(i2++));
          } else if (p1 < p2) {
            sum += bob::math::kullback_leibler_divergence(values_1(i1++), T2(0));
          } else{
            sum += bob::math::kullback_leibler_divergence(T2(0), values_2(i2++));
          }
        }
        // roll up the remaining parts of the histograms
        while (i1 < i1_end) sum += bob::math::kullback_leibler_divergence(values_1(i1++), T2(0));
        while (i2 < i2_end) sum += bob::math::kullback_leibler_divergence(T2(0), values_2(i2++));
        return sum;
      }

  } // namespace math
} // namespace bob

#endif // BOB_MATH_HISTOGRAM_H
