/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 17 Nov 11:42:09 2011 CET
 *
 * @brief Binds the Singular Value Decomposition based on LAPACK to python.
 */

#include <boost/python.hpp>

#include "core/logging.h"
#include "math/svd.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace tm = Torch::math;

static const char* SVD_DOC = "Decompose a matrix A into singular values/vectors A=U*S*V'. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix!";

static void svd(tp::const_ndarray A, tp::ndarray U, tp::ndarray S,
    tp::ndarray V) {
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,1> S_ = S.bz<double,1>();
  blitz::Array<double,2> V_ = V.bz<double,2>();
  tm::svd(A.bz<double,2>(), U_, S_, V_);
}

void bind_math_svd() {
  def("svd", &svd, (arg("A"),arg("U"),arg("S"),arg("V")), SVD_DOC);
}
