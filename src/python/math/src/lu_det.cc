/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the LU Decomposition based on LAPACK into python.
 */

#include <boost/python.hpp>

#include "math/lu_det.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace tm = Torch::math;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* LU_DOC = "Decompose a matrix A into L and U, s.t P*A = L*U. The decomposition is performed using the LAPACK library.";
static const char* DET_DOC = "Compute the determinant of a square matrix. The computation is based on the LU decomposition.";
static const char* INV_DOC = "Compute the inverse of a square matrix. The computation is based on the LU decomposition.";

static void lu(tp::const_ndarray A, tp::ndarray L, 
    tp::ndarray U, tp::ndarray P) {
  blitz::Array<double,2> L_ = L.bz<double,2>();
  blitz::Array<double,2> U_ = U.bz<double,2>();
  blitz::Array<double,2> P_ = P.bz<double,2>();
  tm::lu(A.bz<double,2>(), L_, U_, P_);
}

static void inv(tp::const_ndarray A, tp::ndarray B) {
  blitz::Array<double,2> B_ = B.bz<double,2>();
  tm::inv(A.bz<double,2>(), B_);
}

void bind_math_lu_det()
{
  // LU Decomposition
  def("lu", &lu, (arg("A"), arg("L"), arg("U"), arg("P")), LU_DOC);
  // Compute the determinant of a square matrix, based on an LU decomposition
  def("det", (double (*)(const blitz::Array<double,2>& A))&Torch::math::det, (arg("A")), DET_DOC);
  // Compute the inverse of a square matrix, based on an LU decomposition
  def("inv", &inv, (arg("A"), arg("B")), INV_DOC);
}
