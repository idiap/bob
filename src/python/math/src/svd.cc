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

#include "core/logging.h"
#include "math/svd.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace math = Torch::math;

static const char* SVD_DOC = "Decompose a matrix A into singular values/vectors A=U*S*V'. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix!";

static void svd(tp::const_ndarray A, tp::ndarray U, tp::ndarray S,
    tp::ndarray V) {
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,1> S_ = S.bz<double,1>();
  blitz::Array<double,2> V_ = V.bz<double,2>();
  math::svd(A.bz<double,2>(), U_, S_, V_);
}

void bind_math_svd() {
  def("svd", &svd, (arg("A"),arg("U"),arg("S"),arg("V")), SVD_DOC);
}
