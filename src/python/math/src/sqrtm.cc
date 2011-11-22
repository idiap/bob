/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed 12 Oct 2011
 *
 * Binds the matrix square root for symmetric definite-positive matrices 
 * into python.
 */

#include "core/python/ndarray.h"
#include "math/sqrtm.h"

using namespace boost::python;
namespace math = Torch::math;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* SQRTSYMREAL_DOC = "Returns the square root of a real symmetric positive-definite real matrix!";

static void py_sqrt_symreal(tp::const_ndarray A, tp::ndarray B) {
  const ca::typeinfo& info = A.type();
  if (info.nd != 2 || info.dtype != ca::t_float64) 
    PYTHON_ERROR(TypeError, "method only accepts 2D float64 arrays");
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::sqrtSymReal(A.bz<double,2>(), B_);
}

static void py_sqrt_symreal_(tp::const_ndarray A, tp::ndarray B) {
  const ca::typeinfo& info = A.type();
  if (info.nd != 2 || info.dtype != ca::t_float64) 
    PYTHON_ERROR(TypeError, "method only accepts 2D float64 arrays");
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::sqrtSymReal_(A.bz<double,2>(), B_);
}

static object py_sqrt_symreal_alloc(tp::const_ndarray A) {
  const ca::typeinfo& info = A.type();
  if (info.nd != 2 || info.dtype != ca::t_float64) 
    PYTHON_ERROR(TypeError, "method only accepts 2D float64 arrays");
  tp::ndarray B(ca::t_float64, info.shape[0], info.shape[1]);
  blitz::Array<double,2> B_ = B.bz<double,2>();
  math::sqrtSymReal(A.bz<double,2>(), B_);
  return B.self();
}

void bind_math_sqrtm() {
  def("sqrtSymReal", &py_sqrt_symreal, (arg("A"),arg("B")), SQRTSYMREAL_DOC);
  def("sqrtSymReal_", &py_sqrt_symreal_, (arg("A"),arg("B")), SQRTSYMREAL_DOC);
  def("sqrtSymReal", &py_sqrt_symreal_alloc, (arg("A")), SQRTSYMREAL_DOC);
}
