/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 22 Jun 03:52:01 2011 CEST
 *
 * Binds the Eigenvalue Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/eig.h"
#include "core/python/pycore.h"

using namespace boost::python;
namespace math = Torch::math;
namespace tp = Torch::python;

static const char* EIGSYMREAL_DOC = "Decompose a matrix A into eigenvalues/vectors A=V*D*V-1. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix!";
static const char* EIGSYMGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! A and B should be symmetric and B should be positive definite.";
static const char* EIGGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! Also note both eigen values and vectors are organized in decreasing eigen-value order.";

static void py_eig_symreal(const blitz::Array<double,2>& A,
    numeric::array& V, numeric::array& D) {
  blitz::Array<double,2> V_ = tp::numpy_bz<double,2>(V);
  blitz::Array<double,1> D_ = tp::numpy_bz<double,1>(D);
  math::eigSymReal(A, V_, D_);
}

static tuple py_eig_symreal_alloc(const blitz::Array<double,2>& A) {
  blitz::Array<double,2> V(A.shape());
  blitz::Array<double,1> D(A.extent(0));
  math::eigSymReal(A, V, D);
  return make_tuple(V, D);
}

static void py_eig_sym(const blitz::Array<double,2>& A, 
    const blitz::Array<double,2>& B, numeric::array& V, numeric::array& D) {
  blitz::Array<double,2> V_ = tp::numpy_bz<double,2>(V);
  blitz::Array<double,1> D_ = tp::numpy_bz<double,1>(D);
  math::eigSym(A, B, V_, D_);
}

static tuple py_eig_sym_alloc(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B) {
  blitz::Array<double,2> V(A.shape());
  blitz::Array<double,1> D(A.extent(0));
  math::eigSym(A, B, V, D);
  return make_tuple(V, D);
}

static void py_eig(const blitz::Array<double,2>& A,
    const blitz::Array<double,2>& B, numeric::array& V, numeric::array& D) {
  blitz::Array<double,2> V_ = tp::numpy_bz<double,2>(V);
  blitz::Array<double,1> D_ = tp::numpy_bz<double,1>(D);
  math::eig(A, B, V_, D_);
}

static tuple py_eig_alloc(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B) {
  blitz::Array<double,2> V(A.shape());
  blitz::Array<double,1> D(A.extent(0));
  math::eig(A, B, V, D);
  return make_tuple(V, D);
}

void bind_math_eig() {
  def("eigSymReal", &py_eig_symreal, (arg("A")), EIGSYMREAL_DOC);
  def("eigSymReal", &py_eig_symreal_alloc, (arg("A")), EIGSYMREAL_DOC);
  def("eigSym", &py_eig_sym, (arg("A"),arg("B")), EIGSYMGEN_DOC);
  def("eigSym", &py_eig_sym_alloc, (arg("A"),arg("B")), EIGSYMGEN_DOC);
  def("eig", &py_eig, (arg("A"),arg("B")), EIGGEN_DOC);
  def("eig", &py_eig_alloc, (arg("A"),arg("B")), EIGGEN_DOC);
}
