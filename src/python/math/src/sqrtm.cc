/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed 12 Oct 2011
 *
 * Binds the matrix square root for symmetric definite-positive matrices 
 * into python.
 */

#include <boost/python.hpp>

#include "math/sqrtm.h"

using namespace boost::python;
namespace math = Torch::math;

static const char* SQRTSYMREAL_DOC = "Returns the square root of a real symmetric positive-definite real matrix!";

static object py_sqrt_symreal(const blitz::Array<double,2>& A) {
  blitz::Array<double,2> B(A.extent(0), A.extent(1));
  math::sqrtSymReal(A, B);
  return object(B);
}

void bind_math_sqrtm() {
  def("sqrtSymReal", &Torch::math::sqrtSymReal, (arg("A"),arg("B")), SQRTSYMREAL_DOC);
  def("sqrtSymReal_", &Torch::math::sqrtSymReal_, (arg("A"),arg("B")), SQRTSYMREAL_DOC);
  def("sqrtSymReal", &py_sqrt_symreal, (arg("A")), SQRTSYMREAL_DOC);
}
