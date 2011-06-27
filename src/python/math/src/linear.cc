/**
 * @file src/python/math/src/linear.cc 
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 21 Jun 2011 11:41:56 CEST
 *
 * Binds the basic matrix and vector operations
 */

#include <boost/python.hpp>
#include "math/linear.h"

using namespace boost::python;
namespace math = Torch::math;

static const char* PROD_MATMAT_DOC = "Computes the product of two matrices (2D arrays). Sizes are checked (to avoid the check, use the _ variants).";
static const char* PROD_MATMAT_DOC_ = "Computes the product of two matrices (2D arrays). Sizes are NOT checked.";
static const char* PROD_MATVEC_DOC = "Computes the product between a matrix (2D array) and a vector (1D array). Sizes are checked (to avoid the check, use the _ variants).";
static const char* PROD_MATVEC_DOC_ = "Computes the product between a matrix (2D array) and a vector (1D array). Sizes are NOT checked.";
static const char* OUTER_DOC = "Computes the outer product of two vectors (1D arrays), generating a matrix (2D array). Sizes are checked (to avoid the check, use the _ variants).";
static const char* OUTER_DOC_ = "Computes the outer product of two vectors (1D arrays), generating a matrix (2D array). Sizes are NOT checked.";
static const char* DOT_DOC = "Computes the dot product between two vectors (1D arrays). Sizes are checked (to avoid the check, use the _ variants).";
static const char* DOT_DOC_ = "Computes the dot product between two vectors (1D arrays). Sizes are NOT checked.";
static const char* TRACE_DOC = "Computes the trace of a matrix. Sizes are checked (to avoid the check, use the _ variants).";
static const char* TRACE_DOC_ = "Computes the trace of a matrix. Sizes are NOT checked.";
static const char* NORM_DOC = "Normalizes a vector 'i' and outputs the normalized vector in 'o'. Sizes are checked (to avoid the check, use the _ variants).";
static const char* NORM_DOC_ = "Normalizes a vector 'i' and outputs the normalized vector in 'o'. Sizes are NOT checked.";
static const char* NORM_DOC_SELF = "Normalizes a vector 'i' and outputs the normalized vector vector 'i' (i.e., the same input vector).";
static const char* EUCL_NORM_DOC = "Computes the Euclidean norm of a vector 'i'.";

template<typename T1, typename T2, typename T3> 
static blitz::Array<T3,2> r_prod_mm(const blitz::Array<T1,2>& A, const blitz::Array<T2,2>& B) {
  blitz::Array<T3,2> C(A.extent(0), B.extent(1));
  math::prod(A, B, C); //we are slow already, use the checked version
  return C;
}

template<typename T1, typename T2, typename T3> 
static blitz::Array<T3,1> r_prod_mv(const blitz::Array<T1,2>& A, const blitz::Array<T2,1>& b) {
  blitz::Array<T3,1> c(A.extent(0));
  math::prod(A, b, c); //we are slow already, use the checked version
  return c;
}

template<typename T1, typename T2, typename T3>
static blitz::Array<T3,1> r_prod_vm(const blitz::Array<T1,1>& a, const blitz::Array<T2,2>& B) {
  blitz::Array<T3,1> c(B.extent(1));
  math::prod(a, B, c); //we are slow already, use the checked version
  return c;
}

template<typename T1, typename T2, typename T3> 
static blitz::Array<T3,2> r_outer(const blitz::Array<T1,1>& a, const blitz::Array<T2,1>& b) {
  blitz::Array<T3,2> C(a.extent(0), b.extent(0));
  math::prod(a, b, C); //we are slow already, use the checked version
  return C;
}

template<typename T>
static blitz::Array<T,1> r_normalize(const blitz::Array<T,1>& i) {
  blitz::Array<T,1> o(i.shape());
  math::normalize_(i, o);
  return o;
}

/**
 * This template method simplifies the declaration of python bindings.
 */
template<typename T1, typename T2, typename T3> void def_linear() {

  //matrix-matrix
  def("prod_", (void (*)(const blitz::Array<T1,2>&, const blitz::Array<T2,2>&, blitz::Array<T3,2>&))&Torch::math::prod_, (arg("A"), arg("B"), arg("C")), PROD_MATMAT_DOC_);
  def("prod", (void (*)(const blitz::Array<T1,2>&, const blitz::Array<T2,2>&, blitz::Array<T3,2>&))&Torch::math::prod, (arg("A"), arg("B"), arg("C")), PROD_MATMAT_DOC);
  def("prod", &r_prod_mm<T1,T2,T3>, (arg("A"), arg("B")), PROD_MATMAT_DOC);

  //vector-matrix
  def("prod_", (void (*)(const blitz::Array<T1,1>&, const blitz::Array<T2,2>&, blitz::Array<T3,1>&))&Torch::math::prod_, (arg("a"), arg("B"), arg("c")), PROD_MATVEC_DOC_);
  def("prod", (void (*)(const blitz::Array<T1,1>&, const blitz::Array<T2,2>&, blitz::Array<T3,1>&))&Torch::math::prod, (arg("a"), arg("B"), arg("c")), PROD_MATVEC_DOC);
  def("prod", &r_prod_vm<T1,T2,T3>, (arg("a"), arg("B")), PROD_MATVEC_DOC);

  //matrix-vector
  def("prod_", (void (*)(const blitz::Array<T1,2>&, const blitz::Array<T2,1>&, blitz::Array<T3,1>&))&Torch::math::prod_, (arg("A"), arg("b"), arg("c")), PROD_MATVEC_DOC_);
  def("prod", (void (*)(const blitz::Array<T1,2>&, const blitz::Array<T2,1>&, blitz::Array<T3,1>&))&Torch::math::prod, (arg("A"), arg("b"), arg("c")), PROD_MATVEC_DOC);
  def("prod", &r_prod_mv<T1,T2,T3>, (arg("A"), arg("b")), PROD_MATVEC_DOC);

  //outer product + aliases
  def("prod_", (void (*)(const blitz::Array<T1,1>&, const blitz::Array<T2,1>&, blitz::Array<T3,2>&))&Torch::math::prod_, (arg("a"), arg("b"), arg("C")), OUTER_DOC_);
  def("prod", (void (*)(const blitz::Array<T1,1>&, const blitz::Array<T2,1>&, blitz::Array<T3,2>&))&Torch::math::prod, (arg("a"), arg("b"), arg("C")), OUTER_DOC);
  def("prod", &r_outer<T1,T2,T3>, (arg("a"), arg("b")), OUTER_DOC);
  def("outer_", (void (*)(const blitz::Array<T1,1>&, const blitz::Array<T2,1>&, blitz::Array<T3,2>&))&Torch::math::prod_, (arg("a"), arg("b"), arg("C")), OUTER_DOC_);
  def("outer", (void (*)(const blitz::Array<T1,1>&, const blitz::Array<T2,1>&, blitz::Array<T3,2>&))&Torch::math::prod, (arg("a"), arg("b"), arg("C")), OUTER_DOC);
  def("outer", &r_outer<T1,T2,T3>, (arg("a"), arg("b")), OUTER_DOC);

  //dot product
  def("dot_", &Torch::math::dot_<T1,T2>, (arg("a"), arg("b")), DOT_DOC_);
  def("dot", &Torch::math::dot<T1,T2>, (arg("a"), arg("b")), DOT_DOC);

  //trace
  def("trace_", &Torch::math::trace_<T1>, (arg("A")), TRACE_DOC_);
  def("trace", &Torch::math::trace<T1>, (arg("A")), TRACE_DOC);

  //normalization
  def("normalize_", &Torch::math::normalize_<T1,T2>, (arg("input"),arg("output")), NORM_DOC_);
  def("normalize", &Torch::math::normalize<T1,T2>, (arg("input"),arg("output")), NORM_DOC);
  def("normalize", &r_normalize<T1>, (arg("input")), NORM_DOC);
  def("norm", &Torch::math::norm<T1>, (arg("input")), EUCL_NORM_DOC);

  //self normalization
  def("normalizeSelf", &Torch::math::normalizeSelf<T1>, (arg("input")), NORM_DOC_SELF);

}

void bind_math_linear() {
  def_linear<double, double, double>();
  def_linear<double, float, double>();
  def_linear<float, double, double>();
  def_linear<float, float, float>();
}
