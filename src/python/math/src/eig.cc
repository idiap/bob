/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 22 Jun 03:52:01 2011 CEST
 *
 * Binds the Eigenvalue Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/eig.h"

using namespace boost::python;
namespace math = Torch::math;

static const char* EIGSYMREAL_DOC = "Decompose a matrix A into eigenvalues/vectors A=V*D*V-1. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix!";
static const char* EIGSYMGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! A and B should be symmetric and B should be positive definite.";
static const char* EIGGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! Also note both eigen values and vectors are organized in decreasing eigen-value order.";

static tuple py_eig_symreal(const blitz::Array<double,2>& A) {
  blitz::Array<double,2> V(A.shape());
  blitz::Array<double,1> D(A.extent(0));
  math::eigSymReal(A, V, D);
  return make_tuple(V, D);
}

static tuple py_eig_sym(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B) {
  blitz::Array<double,2> V(A.shape());
  blitz::Array<double,1> D(A.extent(0));
  math::eigSym(A, B, V, D);
  return make_tuple(V, D);
}

static tuple py_eig(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B) {
  blitz::Array<double,2> V(A.shape());
  blitz::Array<double,1> D(A.extent(0));
  math::eig(A, B, V, D);
  return make_tuple(V, D);
}

void bind_math_eig() {
  def("eigSymReal", &Torch::math::eigSymReal, (arg("A"),arg("V"),arg("D")), EIGSYMREAL_DOC);
  def("eigSymReal_", &Torch::math::eigSymReal_, (arg("A"),arg("V"),arg("D")), EIGSYMREAL_DOC);
  def("eigSymReal", &py_eig_symreal, (arg("A")), EIGSYMREAL_DOC);
  def("eigSym", &Torch::math::eigSym, (arg("A"),arg("B"),arg("V"),arg("D")), EIGSYMGEN_DOC);
  def("eigSym_", &Torch::math::eigSym_, (arg("A"),arg("B"),arg("V"),arg("D")), EIGSYMGEN_DOC);
  def("eigSym", &py_eig_sym, (arg("A"),arg("B")), EIGSYMGEN_DOC);
  def("eig", &Torch::math::eig, (arg("A"),arg("B"),arg("V"),arg("D")), EIGGEN_DOC);
  def("eig_", &Torch::math::eig_, (arg("A"),arg("B"),arg("V"),arg("D")), EIGGEN_DOC);
  def("eig", &py_eig, (arg("A"),arg("B")), EIGGEN_DOC);
}
