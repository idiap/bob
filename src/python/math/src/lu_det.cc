/**
 * @file python/math/src/lu_det.cc
 * @date Tue Jun 7 01:00:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the LU Decomposition, determinant and matrix inversion
 *   based on LAPACK into python.
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

#include "math/lu_det.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace math = bob::math;
namespace tp = bob::python;
namespace ca = bob::core::array;

namespace math = bob::math;

static const char* LU_DOC = "Decompose a matrix A into L and U, s.t P*A = L*U. The decomposition is performed using the LAPACK library.";
static const char* LU_P_DOC = "Decompose a matrix A into L and U, s.t P*A = L*U. The decomposition is performed using the LAPACK library. This function will allocate the resulting arrays 'L', 'U' an d 'P' internally every time it is called.";
static const char* DET_DOC = "Compute the determinant of a square matrix. The computation is based on the LU decomposition.";
static const char* INV_DOC = "Compute the inverse of a square matrix. The computation is based on the LU decomposition.";
static const char* INV_P_DOC = "Compute the inverse of a square matrix. The computation is based on the LU decomposition. This function will allocate the resulting array 'B' internally every time it is called.";

static void lu(tp::const_ndarray A, tp::ndarray L, 
    tp::ndarray U, tp::ndarray P) {
  blitz::Array<double,2> L_ = L.bz<double,2>();
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,2> P_ = P.bz<double,2>();
  math::lu(A.bz<double,2>(), L_, U_, P_);
}

static void lu_(tp::const_ndarray A, tp::ndarray L, 
    tp::ndarray U, tp::ndarray P) {
  blitz::Array<double,2> L_ = L.bz<double,2>();
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,2> P_ = P.bz<double,2>();
  math::lu_(A.bz<double,2>(), L_, U_, P_);
}

static tuple py_lu(tp::const_ndarray A) {
  const blitz::Array<double,2> A_ = A.bz<double,2>();
  int M = A_.extent(0);
  int N = A_.extent(1);
  int minMN = std::min(M,N);

  tp::ndarray L(ca::t_float64, M, minMN);
  tp::ndarray U(ca::t_float64, minMN, N);
  tp::ndarray P(ca::t_float64, minMN, minMN);
  blitz::Array<double,2> L_ = L.bz<double,2>();
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,2> P_ = P.bz<double,2>();
  math::lu(A_, L_, U_, P_);
  return make_tuple(L, U, P);
}

static double det(tp::const_ndarray A) {
  return math::det(A.bz<double,2>());
}

static double det_(tp::const_ndarray A) {
  return math::det_(A.bz<double,2>());
}

static void inv(tp::const_ndarray A, tp::ndarray B) {
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::inv(A.bz<double,2>(), B_);
}

static void inv_(tp::const_ndarray A, tp::ndarray B) {
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::inv_(A.bz<double,2>(), B_);
}

static object py_inv(tp::const_ndarray A) {
  const  blitz::Array<double,2> A_ = A.bz<double,2>();
  tp::ndarray B(ca::t_float64, A_.extent(0), A_.extent(1));
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::inv(A_, B_);
  return B.self();
}

void bind_math_lu_det() {
  // LU Decomposition
  def("lu", &lu, (arg("A"), arg("L"), arg("U"), arg("P")), LU_DOC);
  def("lu_", &lu_, (arg("A"), arg("L"), arg("U"), arg("P")), LU_DOC);
  def("lu", &py_lu, (arg("A")), LU_P_DOC);
  // Compute the determinant of a square matrix, based on an LU decomposition
  def("det_", &det_, (arg("A")), DET_DOC);
  def("det", &det, (arg("A")), DET_DOC);
  // Compute the inverse of a square matrix, based on an LU decomposition
  def("inv", &inv, (arg("A"), arg("B")), INV_DOC);
  def("inv_", &inv_, (arg("A"), arg("B")), INV_DOC);
  def("inv", &py_inv, (arg("A")), INV_P_DOC);
}
