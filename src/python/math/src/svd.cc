/**
 * @file python/math/src/svd.cc
 * @date Sat Mar 19 22:14:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Singular Value Decomposition based on LAPACK to python.
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

#include "math/svd.h"
#include <algorithm>

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = bob::python;
namespace math = bob::math;
namespace ca = bob::core::array;

static void svd1(tp::const_ndarray A, tp::ndarray U, tp::ndarray S,
    tp::ndarray V) {
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,1> S_ = S.bz<double,1>();
  blitz::Array<double,2> V_ = V.bz<double,2>();
  math::svd(A.bz<double,2>(), U_, S_, V_);
}

static void svd1_(tp::const_ndarray A, tp::ndarray U, tp::ndarray S,
    tp::ndarray V) {
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,1> S_ = S.bz<double,1>();
  blitz::Array<double,2> V_ = V.bz<double,2>();
  math::svd_(A.bz<double,2>(), U_, S_, V_);
}

static void svd2(tp::const_ndarray A, tp::ndarray U, tp::ndarray S) {
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,1> S_ = S.bz<double,1>();
  bob::math::svd(A.bz<double,2>(), U_, S_);
}

static void svd2_(tp::const_ndarray A, tp::ndarray U, tp::ndarray S) {
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,1> S_ = S.bz<double,1>();
  bob::math::svd_(A.bz<double,2>(), U_, S_);
}

static tuple svd3(tp::const_ndarray A) {
  const ca::typeinfo& info = A.type();
  if (info.nd != 2) 
    PYTHON_ERROR(TypeError, "input matrix has to be 2-dimensional");
  int M = info.shape[0];
  int N = info.shape[1];
  int nb_singular = std::min(M,N);
  tp::ndarray U(ca::t_float64, M, M);
  blitz::Array<double,2> U_ = U.bz<double,2>();
  tp::ndarray S(ca::t_float64, nb_singular);
  blitz::Array<double,1> S_ = S.bz<double,1>();
  tp::ndarray V(ca::t_float64, N, N);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  bob::math::svd_(A.bz<double,2>(), U_, S_, V_);
  return make_tuple(U.self(), S.self(), V.self());
}

void bind_math_svd() {

  static const char* SVD1_DOC = "Decomposes a matrix A into singular values/vectors A=U*S*V'. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix!";
  static const char* SVD2_DOC = "Decomposes a matrix A into singular values/vectors. It only returns the first min(M,N) columns of U (left eigenvectors with associated to non-zero singular values) as well as the singular values in a vector S. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix!";
  static const char* SVD3_DOC = "Decomposes a matrix A into singular values/vectors A=U*S*V'. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix! This version will allocate the resulting arrays 'U', 'S' and 'V' internally every time it is called.";

  def("svd", &svd1, (arg("A"),arg("U"),arg("S"),arg("V")), SVD1_DOC);
  def("svd_", &svd1_, (arg("A"),arg("U"),arg("S"),arg("V")), SVD1_DOC);
  def("svd", &svd2, (arg("A"),arg("U"),arg("S")), SVD2_DOC);
  def("svd_", &svd2_, (arg("A"),arg("U"),arg("S")), SVD2_DOC);
  def("svd", &svd3, (arg("A")), SVD3_DOC);
}
