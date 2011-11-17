/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 22 Jun 03:52:01 2011 CEST
 *
 * Binds the Eigenvalue Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/eig.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace math = Torch::math;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* EIGSYMREAL_DOC = "Decompose a matrix A into eigenvalues/vectors A=V*D*V-1. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix!";
static const char* EIGSYMGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! A and B should be symmetric and B should be positive definite.";
static const char* EIGGEN_DOC = "Find the eigenvalues/vectors decomposition of the following problem: A*X = lambda*B*X. The decomposition is performed using the LAPACK library. The eigenvalues are returned as a 1D array rather than a 2D diagonal matrix! Also note both eigen values and vectors are organized in decreasing eigen-value order.";

static void py_eig_symreal(tp::const_ndarray A, tp::ndarray V, tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSymReal(A.bz<double,2>(), V_, D_);
}

static tuple py_eig_symreal_alloc(tp::const_ndarray A) {
  const ca::typeinfo& info = A.type();
  tp::ndarray V(info);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  tp::ndarray D(info.dtype, info.shape[0]);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSymReal(A.bz<double,2>(), V_, D_);
  return make_tuple(V, D);
}

static void py_eig_sym(tp::const_ndarray A, 
    tp::const_ndarray B, tp::ndarray V, tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSym(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
}

static tuple py_eig_sym_alloc(tp::const_ndarray A, tp::const_ndarray B) {
  const ca::typeinfo& info = A.type();
  tp::ndarray V(info);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  tp::ndarray D(info.dtype, info.shape[0]);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eigSym(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
  return make_tuple(V, D);
}

static void py_eig(tp::const_ndarray A, tp::const_ndarray B, tp::ndarray V,
    tp::ndarray D) {
  blitz::Array<double,2> V_ = V.bz<double,2>();
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eig(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
}

static tuple py_eig_alloc(tp::const_ndarray A, tp::const_ndarray B) {
  const ca::typeinfo& info = A.type();
  tp::ndarray V(info);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  tp::ndarray D(info.dtype, info.shape[0]);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  math::eig(A.bz<double,2>(), B.bz<double,2>(), V_, D_);
  return make_tuple(V, D);
}

void bind_math_eig() {
  def("eigSymReal", &py_eig_symreal, (arg("A"),arg("V"),arg("D")),
      EIGSYMREAL_DOC);
  def("eigSymReal", &py_eig_symreal_alloc, (arg("A")), EIGSYMREAL_DOC);
  def("eigSym", &py_eig_sym, (arg("A"),arg("B"),arg("D")), EIGSYMGEN_DOC);
  def("eigSym", &py_eig_sym_alloc, (arg("A"),arg("B")), EIGSYMGEN_DOC);
  def("eig", &py_eig, (arg("A"),arg("B"),arg("V"),arg("D")), EIGGEN_DOC);
  def("eig", &py_eig_alloc, (arg("A"),arg("B")), EIGGEN_DOC);
}
