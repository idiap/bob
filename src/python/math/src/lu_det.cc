/**
 * @file src/python/math/src/lu_det.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the LU Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/lu_det.h"

using namespace boost::python;

static const char* LU_DOC = "Decompose a matrix A into L and U, s.t P*A = L*U. The decomposition is performed using the LAPACK library.";
static const char* DET_DOC = "Compute the determinant of a square matrix. The computation is based on the LU decomposition.";
static const char* INV_DOC = "Compute the inverse of a square matrix. The computation is based on the LU decomposition.";

void bind_math_lu_det()
{
  // LU Decomposition
  def("lu", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& L, blitz::Array<double,2>& U, blitz::Array<double,2>& P))&Torch::math::lu, (arg("A"), arg("L"), arg("U"), arg("P")), LU_DOC);
  // Compute the determinant of a square matrix, based on an LU decomposition
  def("det", (double (*)(const blitz::Array<double,2>& A))&Torch::math::det, (arg("A")), DET_DOC);
  // Compute the inverse of a square matrix, based on an LU decomposition
  def("inv", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& B))&Torch::math::inv, (arg("A"), arg("B")), INV_DOC);
}

