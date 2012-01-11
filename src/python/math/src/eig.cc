/**
 * @file python/math/src/eig.cc
 * @date Mon May 16 21:45:27 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Binds the Eigenvalue Decomposition based on LAPACK into python.
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

#include "math/eig.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace math = bob::math;
namespace tp = bob::python;
namespace ca = bob::core::array;

static const char* EIGSYMREAL_DOC = "Decompose a matrix A into eigenvalues/vectors A=V*D*V-1. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix!";
static const char* EIGSYMGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! A and B should be symmetric and B should be positive definite.";
static const char* EIGGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! Also note both eigen values and vectors are organized in decreasing eigen-value order.";

static void py_eig_symreal(tp::const_ndarray A, tp::ndarray V, tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSymReal(A.bz<double,2>(), V_, D_);
}

static void py_eig_symreal_(tp::const_ndarray A, tp::ndarray V, tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSymReal_(A.bz<double,2>(), V_, D_);
}

static tuple py_eig_symreal_alloc(tp::const_ndarray A) {
  const ca::typeinfo& info = A.type();
  tp::ndarray V(info);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  tp::ndarray D(info.dtype, info.shape[0]);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSymReal(A.bz<double,2>(), V_, D_);
  return make_tuple(V, D);
}

static void py_eig_sym(tp::const_ndarray A, 
    tp::const_ndarray B, tp::ndarray V, tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSym(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
}

static void py_eig_sym_(tp::const_ndarray A, 
    tp::const_ndarray B, tp::ndarray V, tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSym_(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
}

static tuple py_eig_sym_alloc(tp::const_ndarray A, tp::const_ndarray B) {
  const ca::typeinfo& info = A.type();
  tp::ndarray V(info);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  tp::ndarray D(info.dtype, info.shape[0]);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSym(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
  return make_tuple(V, D);
}

static void py_eig(tp::const_ndarray A, tp::const_ndarray B, tp::ndarray V,
    tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eig(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
}

static void py_eig_(tp::const_ndarray A, tp::const_ndarray B, tp::ndarray V,
    tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eig_(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
}

static tuple py_eig_alloc(tp::const_ndarray A, tp::const_ndarray B) {
  const ca::typeinfo& info = A.type();
  tp::ndarray V(info);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  tp::ndarray D(info.dtype, info.shape[0]);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eig(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
  return make_tuple(V, D);
}

void bind_math_eig() {
  def("eigSymReal", &py_eig_symreal, (arg("A"),arg("V"),arg("D")),
      EIGSYMREAL_DOC);
  def("eigSymReal_", &py_eig_symreal_, (arg("A"),arg("V"),arg("D")),
      EIGSYMREAL_DOC);
  def("eigSymReal", &py_eig_symreal_alloc, (arg("A")), EIGSYMREAL_DOC);
  def("eigSym", &py_eig_sym, (arg("A"),arg("B"),arg("D")), EIGSYMGEN_DOC);
  def("eigSym_", &py_eig_sym_, (arg("A"),arg("B"),arg("D")), EIGSYMGEN_DOC);
  def("eigSym", &py_eig_sym_alloc, (arg("A"),arg("B")), EIGSYMGEN_DOC);
  def("eig", &py_eig, (arg("A"),arg("B"),arg("V"),arg("D")), EIGGEN_DOC);
  def("eig_", &py_eig_, (arg("A"),arg("B"),arg("V"),arg("D")), EIGGEN_DOC);
  def("eig", &py_eig_alloc, (arg("A"),arg("B")), EIGGEN_DOC);
}
