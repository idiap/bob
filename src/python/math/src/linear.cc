/**
 * @file src/python/math/src/linear.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the basic matrix and vector operations
 */

#include <boost/python.hpp>

#include "core/cast.h"
#include "math/linear.h"

using namespace boost::python;

static const char* PROD_MATMAT_DOC = "Compute the product of two matrices.";
static const char* PROD_MATVEC_DOC = "Compute the product of a matrix and a vector.";
static const char* PROD_VECVEC_DOC = "Compute the product of a (column-)vector and a transposed/row-vector and generate a matrix.";


void bind_math_linear()
{
  // Linear system solver
  def("prod", (void (*)(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B, blitz::Array<double,2>& C))&Torch::math::prod, (arg("A"), arg("B"), arg("C")), PROD_MATMAT_DOC);
  def("prod", (void (*)(const blitz::Array<double,2>& A, const blitz::Array<double,1>& b, blitz::Array<double,1>& c))&Torch::math::prod, (arg("A"), arg("b")), PROD_MATVEC_DOC);
  def("prod", (void (*)(const blitz::Array<double,1>& a, const blitz::Array<double,1>& b, blitz::Array<double,2>& C))&Torch::math::prod, (arg("a"), arg("b")), PROD_VECVEC_DOC);
}

