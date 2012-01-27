/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed 12 Oct 2011
 *
 * Binds the matrix square root for symmetric definite-positive matrices 
 * into python.
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

#include "core/python/ndarray.h"
#include "math/sqrtm.h"

using namespace boost::python;
namespace math = bob::math;
namespace tp = bob::python;
namespace ca = bob::core::array;

static const char* SQRTSYMREAL_DOC = "Returns the square root of a real symmetric positive-definite real matrix! This is done by first computing the eigenvalue decomposition with LAPACK.";

static void py_sqrt_symreal(tp::const_ndarray A, tp::ndarray B) {
  const ca::typeinfo& info = A.type();
  if (info.nd != 2 || info.dtype != ca::t_float64) 
    PYTHON_ERROR(TypeError, "method only accepts 2D float64 arrays");
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::sqrtSymReal(A.bz<double,2>(), B_);
}

static void py_sqrt_symreal_(tp::const_ndarray A, tp::ndarray B) {
  const ca::typeinfo& info = A.type();
  if (info.nd != 2 || info.dtype != ca::t_float64) 
    PYTHON_ERROR(TypeError, "method only accepts 2D float64 arrays");
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::sqrtSymReal_(A.bz<double,2>(), B_);
}

static object py_sqrt_symreal_alloc(tp::const_ndarray A) {
  const ca::typeinfo& info = A.type();
  if (info.nd != 2 || info.dtype != ca::t_float64) 
    PYTHON_ERROR(TypeError, "method only accepts 2D float64 arrays");
  tp::ndarray B(ca::t_float64, info.shape[0], info.shape[1]);
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::sqrtSymReal(A.bz<double,2>(), B_);
  return B.self();
}

void bind_math_sqrtm() {
  def("sqrtSymReal", &py_sqrt_symreal, (arg("A"),arg("B")), SQRTSYMREAL_DOC);
  def("sqrtSymReal_", &py_sqrt_symreal_, (arg("A"),arg("B")), SQRTSYMREAL_DOC);
  def("sqrtSymReal", &py_sqrt_symreal_alloc, (arg("A")), SQRTSYMREAL_DOC);
}
