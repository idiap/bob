/**
 * @file math/python/histogram.cc
 * @date Mon Apr 16
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Binds fast versions of some histogram measures
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

#include <boost/python.hpp>
#include "bob/math/histogram.h"
#include "bob/python/ndarray.h"
#include "bob/core/array_exception.h"

static boost::python::object histogram_intersection(bob::python::const_ndarray h1, bob::python::const_ndarray h2){
  switch(h1.type().dtype){
    case bob::core::array::t_uint8:
      return boost::python::object(bob::math::histogram_intersection(h1.bz<uint8_t,1>(), h2.bz<uint8_t, 1>()));
    case bob::core::array::t_uint16:
      return boost::python::object(bob::math::histogram_intersection(h1.bz<uint16_t,1>(), h2.bz<uint16_t, 1>()));
    case bob::core::array::t_int32:
      return boost::python::object(bob::math::histogram_intersection(h1.bz<int32_t,1>(), h2.bz<int32_t, 1>()));
    case bob::core::array::t_int64:
      return boost::python::object(bob::math::histogram_intersection(h1.bz<int64_t,1>(), h2.bz<int64_t, 1>()));
    case bob::core::array::t_float64:
      return boost::python::object(bob::math::histogram_intersection(h1.bz<double,1>(), h2.bz<double, 1>()));
    default:
      PYTHON_ERROR(TypeError, "Histogram intersection currently not implemented for type '%s'", h1.type().str().c_str());
  }
}

template <class T1>
static boost::python::object inner_histogram_intersection_2(bob::python::const_ndarray index_1, bob::python::const_ndarray values_1, bob::python::const_ndarray index_2, bob::python::const_ndarray values_2){
  switch(values_1.type().dtype){
    case bob::core::array::t_uint8:
      return boost::python::object(bob::math::histogram_intersection(index_1.bz<T1,1>(), values_1.bz<uint8_t,1>(), index_2.bz<T1,1>(), values_2.bz<uint8_t,1>()));
    case bob::core::array::t_uint16:
      return boost::python::object(bob::math::histogram_intersection(index_1.bz<T1,1>(), values_1.bz<uint16_t,1>(), index_2.bz<T1,1>(), values_2.bz<uint16_t,1>()));
    case bob::core::array::t_int32:
      return boost::python::object(bob::math::histogram_intersection(index_1.bz<T1,1>(), values_1.bz<int32_t,1>(), index_2.bz<T1,1>(), values_2.bz<int32_t,1>()));
    case bob::core::array::t_int64:
      return boost::python::object(bob::math::histogram_intersection(index_1.bz<T1,1>(), values_1.bz<int64_t,1>(), index_2.bz<T1,1>(), values_2.bz<int64_t,1>()));
    case bob::core::array::t_float64:
      return boost::python::object(bob::math::histogram_intersection(index_1.bz<T1,1>(), values_1.bz<double,1>(), index_2.bz<T1,1>(), values_2.bz<double,1>()));
    default:
      PYTHON_ERROR(TypeError, "Histogram intersection currently not implemented for type '%s'", values_1.type().str().c_str());
  }
}

static boost::python::object histogram_intersection_2(bob::python::const_ndarray index_1, bob::python::const_ndarray values_1, bob::python::const_ndarray index_2, bob::python::const_ndarray values_2){
  assert(index_1.type().dtype == index_2.type().dtype);
  assert(values_1.type().dtype == values_2.type().dtype);
  switch(index_1.type().dtype){
    case bob::core::array::t_uint8:
      return inner_histogram_intersection_2<uint8_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_uint16:
      return inner_histogram_intersection_2<uint16_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_int32:
      return inner_histogram_intersection_2<int32_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_int64:
      return inner_histogram_intersection_2<int64_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_float64:
      return inner_histogram_intersection_2<double>(index_1, values_1, index_2, values_2);
    default:
      PYTHON_ERROR(TypeError, "Histogram index type '%s' is currently not implemented.", index_1.type().str().c_str());
  }
}

static boost::python::object chi_square(bob::python::const_ndarray h1, bob::python::const_ndarray h2){
  switch(h1.type().dtype){
    case bob::core::array::t_uint8:
      return boost::python::object(bob::math::chi_square(h1.bz<uint8_t,1>(), h2.bz<uint8_t, 1>()));
    case bob::core::array::t_uint16:
      return boost::python::object(bob::math::chi_square(h1.bz<uint16_t,1>(), h2.bz<uint16_t, 1>()));
    case bob::core::array::t_int32:
      return boost::python::object(bob::math::chi_square(h1.bz<int32_t,1>(), h2.bz<int32_t, 1>()));
    case bob::core::array::t_int64:
      return boost::python::object(bob::math::chi_square(h1.bz<int64_t,1>(), h2.bz<int64_t, 1>()));
    case bob::core::array::t_float64:
      return boost::python::object(bob::math::chi_square(h1.bz<double,1>(), h2.bz<double, 1>()));
    default:
      PYTHON_ERROR(TypeError, "Chi square currently not implemented for type '%s'", h1.type().str().c_str());
  }
}

template <class T1>
static boost::python::object inner_chi_square_2(bob::python::const_ndarray index_1, bob::python::const_ndarray values_1, bob::python::const_ndarray index_2, bob::python::const_ndarray values_2){
  switch(values_1.type().dtype){
    case bob::core::array::t_uint8:
      return boost::python::object(bob::math::chi_square(index_1.bz<T1,1>(), values_1.bz<uint8_t,1>(), index_2.bz<T1,1>(), values_2.bz<uint8_t,1>()));
    case bob::core::array::t_uint16:
      return boost::python::object(bob::math::chi_square(index_1.bz<T1,1>(), values_1.bz<uint16_t,1>(), index_2.bz<T1,1>(), values_2.bz<uint16_t,1>()));
    case bob::core::array::t_int32:
      return boost::python::object(bob::math::chi_square(index_1.bz<T1,1>(), values_1.bz<int32_t,1>(), index_2.bz<T1,1>(), values_2.bz<int32_t,1>()));
    case bob::core::array::t_int64:
      return boost::python::object(bob::math::chi_square(index_1.bz<T1,1>(), values_1.bz<int64_t,1>(), index_2.bz<T1,1>(), values_2.bz<int64_t,1>()));
    case bob::core::array::t_float64:
      return boost::python::object(bob::math::chi_square(index_1.bz<T1,1>(), values_1.bz<double,1>(), index_2.bz<T1,1>(), values_2.bz<double,1>()));
    default:
      PYTHON_ERROR(TypeError, "Chi square measure currently not implemented for type '%s'", values_1.type().str().c_str());
  }
}

static boost::python::object chi_square_2(bob::python::const_ndarray index_1, bob::python::const_ndarray values_1, bob::python::const_ndarray index_2, bob::python::const_ndarray values_2){
  assert(index_1.type().dtype == index_2.type().dtype);
  assert(values_1.type().dtype == values_2.type().dtype);
  switch(index_1.type().dtype){
    case bob::core::array::t_uint8:
      return inner_chi_square_2<uint8_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_uint16:
      return inner_chi_square_2<uint16_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_int32:
      return inner_chi_square_2<int32_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_int64:
      return inner_chi_square_2<int64_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_float64:
      return inner_chi_square_2<double>(index_1, values_1, index_2, values_2);
    default:
      PYTHON_ERROR(TypeError, "Histogram index type '%s' is currently not implemented.", index_1.type().str().c_str());
  }
}


static double kullback_leibler(bob::python::const_ndarray h1, bob::python::const_ndarray h2){
  switch(h1.type().dtype){
    case bob::core::array::t_uint8:
      return bob::math::kullback_leibler(h1.bz<uint8_t,1>(), h2.bz<uint8_t, 1>());
    case bob::core::array::t_uint16:
      return bob::math::kullback_leibler(h1.bz<uint16_t,1>(), h2.bz<uint16_t, 1>());
    case bob::core::array::t_int32:
      return bob::math::kullback_leibler(h1.bz<int32_t,1>(), h2.bz<int32_t, 1>());
    case bob::core::array::t_int64:
      return bob::math::kullback_leibler(h1.bz<int64_t,1>(), h2.bz<int64_t, 1>());
    case bob::core::array::t_float64:
      return bob::math::kullback_leibler(h1.bz<double,1>(), h2.bz<double, 1>());
    default:
      PYTHON_ERROR(TypeError, "Kullback-Leibler divergence currently not implemented for type '%s'", h1.type().str().c_str());
  }
}

template <class T1>
static double inner_kullback_leibler_2(bob::python::const_ndarray index_1, bob::python::const_ndarray values_1, bob::python::const_ndarray index_2, bob::python::const_ndarray values_2){
  switch(values_1.type().dtype){
    case bob::core::array::t_uint8:
      return bob::math::kullback_leibler(index_1.bz<T1,1>(), values_1.bz<uint8_t,1>(), index_2.bz<T1,1>(), values_2.bz<uint8_t,1>());
    case bob::core::array::t_uint16:
      return bob::math::kullback_leibler(index_1.bz<T1,1>(), values_1.bz<uint16_t,1>(), index_2.bz<T1,1>(), values_2.bz<uint16_t,1>());
    case bob::core::array::t_int32:
      return bob::math::kullback_leibler(index_1.bz<T1,1>(), values_1.bz<int32_t,1>(), index_2.bz<T1,1>(), values_2.bz<int32_t,1>());
    case bob::core::array::t_int64:
      return bob::math::kullback_leibler(index_1.bz<T1,1>(), values_1.bz<int64_t,1>(), index_2.bz<T1,1>(), values_2.bz<int64_t,1>());
    case bob::core::array::t_float64:
      return bob::math::kullback_leibler(index_1.bz<T1,1>(), values_1.bz<double,1>(), index_2.bz<T1,1>(), values_2.bz<double,1>());
    default:
      PYTHON_ERROR(TypeError, "Kullback-Leibler divergence currently not implemented for type '%s'", values_1.type().str().c_str());
  }
}

static double kullback_leibler_2(bob::python::const_ndarray index_1, bob::python::const_ndarray values_1, bob::python::const_ndarray index_2, bob::python::const_ndarray values_2){
  assert(index_1.type().dtype == index_2.type().dtype);
  assert(values_1.type().dtype == values_2.type().dtype);
  switch(index_1.type().dtype){
    case bob::core::array::t_uint8:
      return inner_kullback_leibler_2<uint8_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_uint16:
      return inner_kullback_leibler_2<uint16_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_int32:
      return inner_kullback_leibler_2<int32_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_int64:
      return inner_kullback_leibler_2<int64_t>(index_1, values_1, index_2, values_2);
    case bob::core::array::t_float64:
      return inner_kullback_leibler_2<double>(index_1, values_1, index_2, values_2);
    default:
      PYTHON_ERROR(TypeError, "Histogram index type '%s' is currently not implemented.", index_1.type().str().c_str());
  }
}


void bind_math_histogram()
{
  boost::python::def(
    "histogram_intersection",
    &histogram_intersection,
    (boost::python::arg("h1"), boost::python::arg("h2")),
    "Computes the histogram intersection between the given histograms, which might be of singular dimension only. The histogram intersection defines a similarity measure, so higher values are better."
  );

  boost::python::def(
    "histogram_intersection",
    &histogram_intersection_2,
    (boost::python::arg("index_1"), boost::python::arg("value_1"), boost::python::arg("index_2"), boost::python::arg("value_2")),
    "Computes the histogram intersection between the given sparse histograms (each given by index and value matrix), which might be of singular dimension only. The histogram intersection defines a similarity measure, so higher values are better."
  );

  boost::python::def(
    "chi_square",
    &chi_square,
    (boost::python::arg("h1"), boost::python::arg("h2")),
    "Computes the chi square distance between the given histograms, which might be of singular dimension only. The chi square function is a distance measure, so lower values are better."
  );

  boost::python::def(
    "chi_square",
    &chi_square_2,
    (boost::python::arg("index_1"), boost::python::arg("value_1"), boost::python::arg("index_2"), boost::python::arg("value_2")),
    "Computes the chi square distance between the given sparse histograms (each given by index and value matrix), which might be of singular dimension only. The chi square function is a distance measure, so lower values are better."
  );

  boost::python::def(
    "kullback_leibler",
    &kullback_leibler,
    (boost::python::arg("h1"), boost::python::arg("h2")),
    "Computes the Kullback-Leibler histogram divergence between the given histograms, which might be of singular dimension only. The Kullback-Leibler divergence is a distance measure, so lower values are better."
  );

  boost::python::def(
    "kullback_leibler",
    &kullback_leibler_2,
    (boost::python::arg("index_1"), boost::python::arg("value_1"), boost::python::arg("index_2"), boost::python::arg("value_2")),
    "Computes the Kullback-Leibler histogram divergence between the given sparse histograms (each given by index and value matrix), which might be of singular dimension only. The Kullback-Leibler divergence is a distance measure, so lower values are better."
  );
}
