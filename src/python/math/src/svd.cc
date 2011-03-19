/**
 * @file src/python/math/src/svd.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Singular Value Decomposition based on LAPACK to python.
 */

#include <boost/python.hpp>

#include "core/logging.h"
#include "math/svd.h"

using namespace boost::python;

static const char* SVD_DOC = "Decompose a matrix A into singular values/vectors A=U*S*V'. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix!";

void bind_math_svd()
{
  // SVD
  def("svd", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, blitz::Array<double,1>& S, blitz::Array<double,2>& V))&Torch::math::svd, (arg("A"),arg("U"),arg("S"),arg("V")), SVD_DOC);
}

