/**
 * @file src/python/math/src/linsolve.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Linear System solver based on LAPACK to python.
 */

#include <boost/python.hpp>

#include "math/linsolve.h"

using namespace boost::python;

static const char* LINSOLVE_DOC = "Solve the linear system A*x=b and return the result as a blitz array. The solver is from the LAPACK library.";
static const char* LINSOLVE_SYMPOS_DOC = "Solve the linear system A*x=b, where A is symmetric definite positive, and return the result as a blitz array. The solver is from the LAPACK library.";

static blitz::Array<double,1> script_linsolve(const blitz::Array<double,2>& A, const blitz::Array<double,1>& b)
{
  blitz::Array<double,1> res(b.extent(0));
  Torch::math::linsolve( A, res, b);
  return res;
}

static blitz::Array<double,1> script_linsolveSympos(const blitz::Array<double,2>& A, const blitz::Array<double,1>& b)
{
  blitz::Array<double,1> res(b.extent(0));
  Torch::math::linsolveSympos( A, res, b);
  return res;
}


void bind_math_linsolve()
{
  // Linear system solver
  def("linsolve", (blitz::Array<double,1> (*)(const blitz::Array<double,2>& A,const blitz::Array<double,1>& b))&script_linsolve, (arg("A"), arg("b")), LINSOLVE_DOC);
  def("linsolve", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b))&Torch::math::linsolve, (arg("A"),arg("output"),arg("b")), LINSOLVE_DOC);
  def("linsolveSympos", (blitz::Array<double,1> (*)(const blitz::Array<double,2>& A,const blitz::Array<double,1>& b))&script_linsolveSympos, (arg("A"), arg("b")), LINSOLVE_SYMPOS_DOC);
  def("linsolveSympos", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,1>& res, const blitz::Array<double,1>& b))&Torch::math::linsolveSympos, (arg("A"),arg("output"),arg("b")), LINSOLVE_SYMPOS_DOC);
}

