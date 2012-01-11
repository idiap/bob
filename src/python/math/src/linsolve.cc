/**
 * @file python/math/src/linsolve.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Linear System solver based on LAPACK to python.
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

#include "math/linsolve.h"
#include "math/cgsolve.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;

static const char* LINSOLVE_DOC = "Solve the linear system A*x=b and return the result as a blitz array. The solver is from the LAPACK library.";
static const char* LINSOLVE_SYMPOS_DOC = "Solve the linear system A*x=b, where A is symmetric definite positive, and return the result as a blitz array. The solver is from the LAPACK library.";
static const char* CGSOLVE_SYMPOS_DOC = "Solve the linear system A*x=b via conjugate gradients, where A is symmetric definite positive, and return the result as a blitz array.";

static object script_linsolve(tp::const_ndarray A, tp::const_ndarray b) {
  const ca::typeinfo& info = A.type();
  tp::ndarray res(info.dtype, info.shape[0]);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  bob::math::linsolve(A.bz<double,2>(), res_, 
      b.bz<double,1>());
  return res.self();
}

static object script_linsolveSympos(tp::const_ndarray A, tp::const_ndarray b) {
  const ca::typeinfo& info = b.type();
  tp::ndarray res(info.dtype, info.shape[0]);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  bob::math::linsolveSympos(A.bz<double,2>(), res_,
      b.bz<double,1>());
  return res.self();
}

static object script_cgsolveSympos(tp::const_ndarray A, tp::const_ndarray b,
    const double acc, const int max_iter) {
  const ca::typeinfo& info = b.type();
  tp::ndarray res(info.dtype, info.shape[0]);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  bob::math::cgsolveSympos(A.bz<double,2>(), res_, 
      b.bz<double,1>(), acc, max_iter);
  return res.self();
}

void bind_math_linsolve()
{
  // Linear system solver -- internal allocation of result
  def("linsolve", &script_linsolve, (arg("A"), arg("b")), LINSOLVE_DOC);
  def("linsolveSympos", &script_linsolveSympos, (arg("A"), arg("b")), LINSOLVE_SYMPOS_DOC);
  def("cgsolveSympos", &script_cgsolveSympos, (arg("A"), arg("b"), arg("acc"), arg("max_iter")), CGSOLVE_SYMPOS_DOC);
  
  def("linsolve", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b))&bob::math::linsolve, (arg("A"),arg("output"),arg("b")), LINSOLVE_DOC);
  def("linsolveSympos", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b))&bob::math::linsolveSympos, (arg("A"),arg("output"),arg("b")), LINSOLVE_SYMPOS_DOC);
  def("cgsolveSympos", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b, const double acc, const int max_iter))&bob::math::linsolveSympos, (arg("A"), arg("output"), arg("b"), arg("acc"), arg("max_iter")), CGSOLVE_SYMPOS_DOC);
}

