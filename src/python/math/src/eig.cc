/**
 * @file src/python/math/src/eig.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Eigenvalue Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/eig.h"

using namespace boost::python;

static const char* EIGSYMREAL_DOC = "Decompose a matrix A into eigenvalues/vectors A=V*D*V-1. The decomposition is performed by the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix!";

void bind_math_eig()
{
  // Eigenvalue Decomposition
  def("eigSymReal", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& V, blitz::Array<double,1>& D))&Torch::math::eigSymReal, (arg("A"),arg("V"),arg("D")), EIGSYMREAL_DOC);
}

