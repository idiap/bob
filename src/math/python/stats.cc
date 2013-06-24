/**
 * @file math/python/stats.cc
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
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
#include <boost/format.hpp>
#include "bob/math/stats.h"
#include "bob/python/ndarray.h"

using namespace boost::python;

static const char* SCATTER_DOC1 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). The resulting matrix 's' has to be square with extents equal to the number of rows in a.";

static const char* SCATTER_DOC2 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). This variant also returns the sample means in 'm'. The resulting arrays 'm' and 's' have to have the correct sizes (s should be square with extents equal to the number of rows in a and m should be a 1D vector with extents equal to the number of rows in a).";

static const char* SCATTER_DOC3 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). This variant returns the sample means and the scatter matrix in a tuple. If you are looking for efficiency, prefer the variants that receive the output variable as one of the input parameters. This version will allocate the resulting arrays 'm' and 's' internally every time it is called.";

template <typename T> static tuple scatter_inner(bob::python::const_ndarray A) {
  const bob::core::array::typeinfo& info = A.type();
  bob::python::ndarray S(info.dtype, info.shape[0], info.shape[0]);
  blitz::Array<T,2> S_ = S.bz<T,2>();
  bob::python::ndarray M(info.dtype, info.shape[0]);
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatter(A.bz<T,2>(), S_, M_);
  return make_tuple(S,M);
}

static tuple scatter(bob::python::const_ndarray A) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_inner<float>(A);
    case bob::core::array::t_float64:
      return scatter_inner<double>(A);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_nocheck_inner(bob::python::const_ndarray A, bob::python::ndarray S) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  bob::math::scatter_<T>(A.bz<T,2>(), S_);
}

static void scatter_nocheck(bob::python::const_ndarray A, bob::python::ndarray S) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_nocheck_inner<float>(A, S);
    case bob::core::array::t_float64:
      return scatter_nocheck_inner<double>(A, S);
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_check_inner(bob::python::const_ndarray A, bob::python::ndarray S) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  bob::math::scatter<T>(A.bz<T,2>(), S_);
}

static void scatter_check(bob::python::const_ndarray A, bob::python::ndarray S) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_check_inner<float>(A, S);
    case bob::core::array::t_float64:
      return scatter_check_inner<double>(A, S);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_M_nocheck_inner(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatter_<T>(A.bz<T,2>(), S_, M_);
}

static void scatter_M_nocheck(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_M_nocheck_inner<float>(A, S, M);
    case bob::core::array::t_float64:
      return scatter_M_nocheck_inner<double>(A, S, M);
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_M_check_inner(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatter<T>(A.bz<T,2>(), S_, M_);
}

static void scatter_M_check(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_M_check_inner<float>(A, S, M);
    case bob::core::array::t_float64:
      return scatter_M_check_inner<double>(A, S, M);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

void bind_math_stats() {
  def("scatter_", &scatter_nocheck, (arg("a"), arg("s")), SCATTER_DOC1);
  def("scatter", &scatter_check, (arg("a"), arg("s")), SCATTER_DOC1);
  
  def("scatter_", &scatter_M_nocheck, (arg("a"), arg("s"), arg("m")), SCATTER_DOC2);
  def("scatter", &scatter_M_check, (arg("a"), arg("s"), arg("m")), SCATTER_DOC2);

  def("scatter", &scatter, (arg("a")), SCATTER_DOC3);

}
