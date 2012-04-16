/**
 * @file python/math/src/histogram.cc
 * @date Mon Apr 16
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Binds fast versions of some histogram measures
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

#include <boost/python.hpp>
#include <math/histogram.h>
#include <core/python/ndarray.h>
#include <core/array_exception.h>

static boost::python::object histogram_intersection(bob::python::const_ndarray h1, bob::python::const_ndarray h2){
  switch(h1.type().dtype){
    case bob::core::array::t_int32:
      return boost::python::object(bob::math::histogram_intersection<int32_t>(h1.bz<int32_t,1>(), h2.bz<int32_t, 1>()));
    case bob::core::array::t_int64:
      return boost::python::object(bob::math::histogram_intersection<int64_t>(h1.bz<int64_t,1>(), h2.bz<int64_t, 1>()));
    case bob::core::array::t_float64:
      return boost::python::object(bob::math::histogram_intersection<double>(h1.bz<double,1>(), h2.bz<double, 1>()));
    default:
      PYTHON_ERROR(TypeError, "Histogram intersection currently not implemented for type '%s'", h1.type().str().c_str());
  }
}

static boost::python::object chi_square(bob::python::const_ndarray h1, bob::python::const_ndarray h2){
  switch(h1.type().dtype){
    case bob::core::array::t_int32:
      return boost::python::object(bob::math::chi_square<int32_t>(h1.bz<int32_t,1>(), h2.bz<int32_t, 1>()));
    case bob::core::array::t_int64:
      return boost::python::object(bob::math::chi_square<int64_t>(h1.bz<int64_t,1>(), h2.bz<int64_t, 1>()));
    case bob::core::array::t_float64:
      return boost::python::object(bob::math::chi_square<double>(h1.bz<double,1>(), h2.bz<double, 1>()));
    default:
      PYTHON_ERROR(TypeError, "Chi square currently not implemented for type '%s'", h1.type().str().c_str());
  }
}


void bind_math_histogram()
{
  boost::python::def(
    "histogram_intersection",
    &histogram_intersection,
    (boost::python::arg("h1"), boost::python::arg("h2")),
    "Computes the histogram intersection between the given histograms, which might be of singular dimension only."
  );

  boost::python::def(
    "chi_square",
    &chi_square,
    (boost::python::arg("h1"), boost::python::arg("h2")),
    "Computes the histogram intersection between the given histograms, which might be of singular dimension only."
  );
}
