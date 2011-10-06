/**
 * @file src/python/math/src/lu_det.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the LU Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/lu_det.h"
#include <algorithm>

using namespace boost::python;

namespace math = Torch::math;

static const char* LU_DOC = "Decompose a matrix A into L and U, s.t P*A = L*U. The decomposition is performed using the LAPACK library.";
static const char* LU_P_DOC = "Decompose a matrix A into L and U, s.t P*A = L*U. The decomposition is performed using the LAPACK library. This function will allocate the resulting arrays 'L', 'U' an d 'P' internally every time it is called.";
static const char* DET_DOC = "Compute the determinant of a square matrix. The computation is based on the LU decomposition.";
static const char* INV_DOC = "Compute the inverse of a square matrix. The computation is based on the LU decomposition.";
static const char* INV_P_DOC = "Compute the inverse of a square matrix. The computation is based on the LU decomposition. This function will allocate the resulting array 'B' internally every time it is called.";

static tuple py_lu(const blitz::Array<double,2>& A) {
  int M = A.extent(0);
  int N = A.extent(1);
  int minMN = std::min(M,N);
  const blitz::TinyVector<int,2> shapeL(M,minMN);
  const blitz::TinyVector<int,2> shapeU(minMN,N);
  const blitz::TinyVector<int,2> shapeP(minMN,minMN);

  blitz::Array<double,2> L(shapeL);
  blitz::Array<double,2> U(shapeU);
  blitz::Array<double,2> P(shapeP);
  math::lu(A, L, U, P);
  return make_tuple(L, U, P);
}

static object py_inv(const blitz::Array<double,2>& A) {
  blitz::Array<double,2> B(A.shape());
  math::inv(A, B);
  return object(B);
}


void bind_math_lu_det()
{
  // LU Decomposition
  def("lu", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& L, blitz::Array<double,2>& U, blitz::Array<double,2>& P))&Torch::math::lu, (arg("A"), arg("L"), arg("U"), arg("P")), LU_DOC);
  def("lu_", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& L, blitz::Array<double,2>& U, blitz::Array<double,2>& P))&Torch::math::lu, (arg("A"), arg("L"), arg("U"), arg("P")), LU_DOC);
  def("lu", &py_lu, (arg("A")), LU_P_DOC);
  // Compute the determinant of a square matrix, based on an LU decomposition
  def("det_", (double (*)(const blitz::Array<double,2>& A))&Torch::math::det, (arg("A")), DET_DOC);
  def("det", (double (*)(const blitz::Array<double,2>& A))&Torch::math::det, (arg("A")), DET_DOC);
  // Compute the inverse of a square matrix, based on an LU decomposition
  def("inv", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& B))&Torch::math::inv, (arg("A"), arg("B")), INV_DOC);
  def("inv_", (void (*)(const blitz::Array<double,2>& A, blitz::Array<double,2>& B))&Torch::math::inv, (arg("A"), arg("B")), INV_DOC);
  def("inv", &py_inv, (arg("A")), INV_P_DOC);
}

