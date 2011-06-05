/**
 * @file src/python/math/src/eig.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Eigenvalue Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/eig.h"

using namespace boost::python;

static const char* EIGSYMREAL_DOC = "Decompose a matrix A into eigenvalues/vectors A=V*D*V-1. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix!";
static const char* EIGSYMGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! A and B should be symmetric and B should be positive definite.";
static const char* EIGGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix!";

void bind_math_eig()
{
  // Eigenvalue Decomposition
  def("eigSymReal", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& V, blitz::Array<double,1>& D))&Torch::math::eigSymReal, (arg("A"),arg("V"),arg("D")), EIGSYMREAL_DOC);
  def("eigSym", (void (*)(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B, blitz::Array<double,2>& V, blitz::Array<double,1>& D))&Torch::math::eigSym, (arg("A"),arg("B"),arg("V"),arg("D")), EIGSYMGEN_DOC);
  def("eig", (void (*)(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B, blitz::Array<double,2>& V, blitz::Array<double,1>& D))&Torch::math::eig, (arg("A"),arg("B"),arg("V"),arg("D")), EIGGEN_DOC);
}

