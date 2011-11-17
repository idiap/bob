/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Linear System solver based on LAPACK to python.
 */

#include "math/linsolve.h"
#include "math/cgsolve.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* LINSOLVE_DOC = "Solve the linear system A*x=b and return the result as a blitz array. The solver is from the LAPACK library.";
static const char* LINSOLVE_SYMPOS_DOC = "Solve the linear system A*x=b, where A is symmetric definite positive, and return the result as a blitz array. The solver is from the LAPACK library.";
static const char* CGSOLVE_SYMPOS_DOC = "Solve the linear system A*x=b via conjugate gradients, where A is symmetric definite positive, and return the result as a blitz array.";

static object script_linsolve(tp::const_ndarray A, tp::const_ndarray b) {
  const ca::typeinfo& info = A.type();
  tp::ndarray res(info.dtype, info.shape[0]);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  Torch::math::linsolve(A.bz<double,2>(), res_, 
      b.bz<double,1>());
  return res.self();
}

static object script_linsolveSympos(tp::const_ndarray A, tp::const_ndarray b) {
  const ca::typeinfo& info = b.type();
  tp::ndarray res(info.dtype, info.shape[0]);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  Torch::math::linsolveSympos(A.bz<double,2>(), res_,
      b.bz<double,1>());
  return res.self();
}

static object script_cgsolveSympos(tp::const_ndarray A, tp::const_ndarray b,
    const double acc, const int max_iter) {
  const ca::typeinfo& info = b.type();
  tp::ndarray res(info.dtype, info.shape[0]);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  Torch::math::cgsolveSympos(A.bz<double,2>(), res_, 
      b.bz<double,1>(), acc, max_iter);
  return res.self();
}

void bind_math_linsolve()
{
  // Linear system solver -- internal allocation of result
  def("linsolve", &script_linsolve, (arg("A"), arg("b")), LINSOLVE_DOC);
  def("linsolveSympos", &script_linsolveSympos, (arg("A"), arg("b")), LINSOLVE_SYMPOS_DOC);
  def("cgsolveSympos", &script_cgsolveSympos, (arg("A"), arg("b"), arg("acc"), arg("max_iter")), CGSOLVE_SYMPOS_DOC);
  
  def("linsolve", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b))&Torch::math::linsolve, (arg("A"),arg("output"),arg("b")), LINSOLVE_DOC);
  def("linsolveSympos", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b))&Torch::math::linsolveSympos, (arg("A"),arg("output"),arg("b")), LINSOLVE_SYMPOS_DOC);
  def("cgsolveSympos", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b, const double acc, const int max_iter))&Torch::math::linsolveSympos, (arg("A"), arg("output"), arg("b"), arg("acc"), arg("max_iter")), CGSOLVE_SYMPOS_DOC);
}

