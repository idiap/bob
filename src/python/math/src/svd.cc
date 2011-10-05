/**
 * @file src/python/math/src/svd.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Singular Value Decomposition based on LAPACK to python.
 */

#include <boost/python.hpp>

#include "math/svd.h"

using namespace boost::python;

static const char* SVD1_DOC = "Decomposes a matrix A into singular values/vectors A=U*S*V'. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix!";
static const char* SVD_1_DOC = "Decomposes a matrix A into singular values/vectors A=U*S*V'. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix! NO checks are performed on the argument arrays.";
static const char* SVD2_DOC = "Decomposes a matrix A into singular values/vectors. It only returns the first min(M,N) columns of U (left eigenvectors with associated to non-zero singular values) as well as the singular values in a vector S. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix!";
static const char* SVD_2_DOC = "Decomposes a matrix A into singular values/vectors. It only returns the first min(M,N) columns of U (left eigenvectors with associated to non-zero singular values) as well as the singular values in a vector S. The decomposition is performed by the LAPACK library. The singular values are returned as a 1D array rather than a 2D diagonal matrix! NO checks are performed on the argument arrays.";

void bind_math_svd()
{
  // SVD
  def("svd", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, blitz::Array<double,1>& S, blitz::Array<double,2>& V))&Torch::math::svd, (arg("A"),arg("U"),arg("S"),arg("V")), SVD1_DOC);
  def("svd_", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, blitz::Array<double,1>& S, blitz::Array<double,2>& V))&Torch::math::svd_, (arg("A"),arg("U"),arg("S"),arg("V")), SVD_1_DOC);
  def("svd", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, blitz::Array<double,1>& S))&Torch::math::svd, (arg("A"),arg("U"),arg("S")), SVD2_DOC);
  def("svd_", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& U, blitz::Array<double,1>& S))&Torch::math::svd_, (arg("A"),arg("U"),arg("S")), SVD_2_DOC);
}
