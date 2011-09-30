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

static const char* NORM_DOC = "Normalizes a vector 'i' and outputs the normalized vector in 'o'. Sizes are checked (to avoid the check, use the _ variants).";
static const char* NORM_DOC_ = "Normalizes a vector 'i' and outputs the normalized vector in 'o'. Sizes are NOT checked.";
static const char* NORM_DOC_SELF = "Normalizes a vector 'i' and outputs the normalized vector vector 'i' (i.e., the same input vector).";
static const char* EUCL_NORM_DOC = "Computes the Euclidean norm of a vector 'i'.";
static const char* EYE_DOC = "Generates an eye 2D matrix. Sizes are checked (to avoid the check, use the _ variants).";
static const char* EYE_DOC_ = "Generates an eye 2D matrix. Sizes are NOT checked.";

template<typename T>
static blitz::Array<T,1> r_normalize(const blitz::Array<T,1>& i) {
  blitz::Array<T,1> o(i.shape());
  math::normalize_(i, o);
  return o;
}
/*
template<typename T>
static blitz::Array<T,2> r1_eye(const int m) {
  blitz::Array<T,2> o(m,m);
  math::eye_(o);
  return o;
}

template<typename T>
static blitz::Array<T,2> r2_eye(const int m, const int n) {
  blitz::Array<T,2> o(m,n);
  math::eye_(o);
  return o;
}
*/

/**
 * This template method simplifies the declaration of python bindings.
 */
template<typename T1, typename T2, typename T3> void def_linear() {

  //normalization
  def("normalize_", &Torch::math::normalize_<T1,T2>, (arg("input"),arg("output")), NORM_DOC_);
  def("normalize", &Torch::math::normalize<T1,T2>, (arg("input"),arg("output")), NORM_DOC);
  def("normalize", &r_normalize<T1>, (arg("input")), NORM_DOC);
  def("norm", &Torch::math::norm<T1>, (arg("input")), EUCL_NORM_DOC);

  //self normalization
  def("normalizeSelf", &Torch::math::normalizeSelf<T1>, (arg("input")), NORM_DOC_SELF);

  //eye
  def("eye_", (void (*)(blitz::Array<T1,2>&))&Torch::math::eye_, (arg("A")), EYE_DOC_);
  def("eye", (void (*)(blitz::Array<T1,2>&))&Torch::math::eye, (arg("A")), EYE_DOC);
  //def("eye", &r1_eye<T1>, (arg("A")), EYE_DOC); // TODO: pass type
}

void bind_math_linear() {
  def_linear<double, double, double>();
  def_linear<double, float, double>();
  def_linear<float, double, double>();
  def_linear<float, float, float>();
}
