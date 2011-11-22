/**
 * @file python/math/src/stats.cc
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "math/stats.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace math = Torch::math;
namespace ca = Torch::core::array;

static const char* SCATTER_DOC1 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). The resulting matrix 'S' has to be square with extents equal to the number of rows in A.";

static const char* SCATTER_DOC2 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). This variant also returns the sample means in 'M'. The resulting arrays 'M' and 'S' have to have the correct sizes (S should be square with extents equal to the number of rows in A and M should be a 1D vector with extents equal to the number of rows in A).";

static const char* SCATTER_DOC3 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). This variant returns the sample means and the scatter matrix in a tuple. If you are looking for efficiency, prefer the variants that receive the output variable as one of the input parameters. This version will allocate the resulting arrays 'M' and 'S' internally every time it is called.";

template <typename T> static tuple scatter_inner(tp::const_ndarray A) {
  const ca::typeinfo& info = A.type();
  tp::ndarray S(info.dtype, info.shape[0], info.shape[0]);
  blitz::Array<T,2> S_ = S.bz<T,2>();
  tp::ndarray M(info.dtype, info.shape[0]);
  blitz::Array<T,1> M_ = M.bz<T,1>();
  math::scatter(A.bz<T,2>(), S_, M_);
  return make_tuple(S,M);
}

static tuple scatter(tp::const_ndarray A) {
  const ca::typeinfo& info = A.type();
  switch (info.dtype) {
    case ca::t_float32:
      return scatter_inner<float>(A);
    case ca::t_float64:
      return scatter_inner<double>(A);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_nocheck_inner(tp::const_ndarray A, tp::ndarray S) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  math::scatter_<T>(A.bz<T,2>(), S_);
}

static void scatter_nocheck(tp::const_ndarray A, tp::ndarray S) {
  const ca::typeinfo& info = A.type();
  switch (info.dtype) {
    case ca::t_float32:
      return scatter_nocheck_inner<float>(A, S);
    case ca::t_float64:
      return scatter_nocheck_inner<double>(A, S);
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_check_inner(tp::const_ndarray A, tp::ndarray S) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  math::scatter<T>(A.bz<T,2>(), S_);
}

static void scatter_check(tp::const_ndarray A, tp::ndarray S) {
  const ca::typeinfo& info = A.type();
  switch (info.dtype) {
    case ca::t_float32:
      return scatter_check_inner<float>(A, S);
    case ca::t_float64:
      return scatter_check_inner<double>(A, S);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_M_nocheck_inner(tp::const_ndarray A, tp::ndarray S,
    tp::ndarray M) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  math::scatter_<T>(A.bz<T,2>(), S_, M_);
}

static void scatter_M_nocheck(tp::const_ndarray A, tp::ndarray S,
    tp::ndarray M) {
  const ca::typeinfo& info = A.type();
  switch (info.dtype) {
    case ca::t_float32:
      return scatter_M_nocheck_inner<float>(A, S, M);
    case ca::t_float64:
      return scatter_M_nocheck_inner<double>(A, S, M);
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_M_check_inner(tp::const_ndarray A, tp::ndarray S,
    tp::ndarray M) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  math::scatter<T>(A.bz<T,2>(), S_, M_);
}

static void scatter_M_check(tp::const_ndarray A, tp::ndarray S,
    tp::ndarray M) {
  const ca::typeinfo& info = A.type();
  switch (info.dtype) {
    case ca::t_float32:
      return scatter_M_check_inner<float>(A, S, M);
    case ca::t_float64:
      return scatter_M_check_inner<double>(A, S, M);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

void bind_math_stats() {
  def("scatter_", &scatter_nocheck, (arg("A"), arg("S")), SCATTER_DOC1);
  def("scatter", &scatter_check, (arg("A"), arg("S")), SCATTER_DOC1);
  
  def("scatter_", &scatter_M_nocheck, (arg("A"), arg("S"), arg("M")), SCATTER_DOC2);
  def("scatter", &scatter_M_check, (arg("A"), arg("S"), arg("M")), SCATTER_DOC2);

  def("scatter", &scatter, (arg("A")), SCATTER_DOC3);

}
