/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 12 Mar 21:48:27 2011 
 *
 * @brief Declares all possible array math operators
 */

#ifndef TORCH_PYTHON_CORE_ARRAY_MATH_H 
#define TORCH_PYTHON_CORE_ARRAY_MATH_H

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"

namespace Torch { namespace python {

  template <typename T, int N> blitz::Array<T,N> acos(blitz::Array<T,N>& i) { 
    return blitz::Array<T,N>(blitz::acos(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> asin(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::asin(i));
  }

  template <typename T, int N> blitz::Array<T,N> atan(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::atan(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> cos(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::cos(i));
  }

  template <typename T, int N> blitz::Array<T,N> cosh(blitz::Array<T,N>& i) { 
    return blitz::Array<T,N>(blitz::cosh(i));
  }

  template <typename T, int N> blitz::Array<T,N> acosh(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::acosh(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> log(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::log(i));
  }

  template <typename T, int N> blitz::Array<T,N> log10(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::log10(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> sin(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::sin(i));
  }

  template <typename T, int N> blitz::Array<T,N> sinh(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::sinh(i));
  }

  template <typename T, int N> blitz::Array<T,N> sqr(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::sqr(i));
  }

  template <typename T, int N> blitz::Array<T,N> sqrt(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::sqrt(i));
  }

  template <typename T, int N> blitz::Array<T,N> tan(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::tan(i));
  }

  template <typename T, int N> blitz::Array<T,N> tanh(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::tanh(i)); 
  }
 
  template <typename T, int N> blitz::Array<T,N> atanh(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::atanh(i)); 
  }
 
  template <typename T, int N> blitz::Array<T,N> cbrt(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::cbrt(i)); 
  }
 
  template <typename T, int N> blitz::Array<T,N> exp(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::exp(i)); 
  }
 
  template <typename T, int N> blitz::Array<T,N> expm1(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::expm1(i)); 
  }
 
  template <typename T, int N> blitz::Array<T,N> erf(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::erf(i)); 
  }
  
  template <typename T, int N> blitz::Array<T,N> erfc(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::erfc(i)); 
  }
 
  template <typename T, int N> blitz::Array<T,N> ilogb(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::ilogb(i)); 
  }
  
  //TODO: template <typename T, int N> blitz::Array<int,N> isnan(blitz::Array<T,N& i) {
  //return blitz::Array<int,N>(blitz::blitz_isnan(i)); 
  //}

  template <typename T, int N> blitz::Array<T,N> j0(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::j0(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> j1(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::j1(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> lgamma(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::lgamma(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> log1p(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::log1p(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> rint(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::rint(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> y0(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::y0(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> y1(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::y1(i)); 
  }

  //operate on floats
  template <typename T, int N> blitz::Array<T,N> ceil(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::ceil(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> floor(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::floor(i)); 
  }

  //operate on complex T
  template <typename T, int N> blitz::Array<T,N> arg(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::arg(i)); 
  }

  template <typename T, int N> blitz::Array<T,N> conj(blitz::Array<T,N>& i) {
    return blitz::Array<T,N>(blitz::conj(i)); 
  }

  template <typename T, int N> void fill(blitz::Array<T,N>& i, const T& v) {
    i = v; 
  }

  template <typename T, int N> void zeroes(blitz::Array<T,N>& i) {
    i = 0; 
  }

  template <typename T, int N> void ones(blitz::Array<T,N>& i) {
    i = 1; 
  }

  template <typename T, int N> blitz::Array<T,N> atan2(blitz::Array<T,N>& x,
      blitz::Array<T,N>& y) {
    return blitz::Array<T,N>(blitz::atan2(x, y));
  }

  template <typename T, int N> blitz::Array<T,N> radius(blitz::Array<T,N>& x,
      blitz::Array<T,N>& y) {
    return blitz::Array<T,N>(blitz::sqrt(blitz::pow2(x) + blitz::pow2(y)));
  }

  /**
   * Methods that operate on integer, float and complex arrays
   */
  template <typename T, int N>
    void bind_non_bool_math (Torch::python::array<T,N>& array) {
      array.object()->def("zeroes", &zeroes<T,N>, "Fills the array with zeroes");
      array.object()->def("ones", &ones<T,N>, "Fills the array with ones");
      array.object()->def("cos", &cos<T,N>, "Cosine, element-wise");
      array.object()->def("cosh", &cosh<T,N>, "Hyperbolic cosine, element-wise");
      array.object()->def("log", &log<T,N>, "Natural logarithm, element-wise");
      array.object()->def("log10", &log10<T,N>, "Base 10 logarithm, element-wise");
      array.object()->def("sin", &sin<T,N>, "Sine, element-wise");
      array.object()->def("sinh", &sinh<T,N>, "Hyperbolic sine, element-wise");
      array.object()->def("sqr", &sqr<T,N>, "self ** 2, element-wise");
      array.object()->def("sqrt", &sqrt<T,N>, "self ** 0.5, element-wise");
      array.object()->def("tan", &tan<T,N>, "Tangent, element-wise");
      array.object()->def("tanh", &tanh<T,N>, "Hyperbolic tangent, element-wise");
      //TODO: array.object()->def("isnan", &isnan<T,N>, "Returns a nonzero integer if the parameter is NaNQ or NaNS (quiet or signalling Not a Number), element-wise.");
    }

  /**
   * Methods that operate on everything that is not complex
   */
  template <typename T, int N>
    void bind_non_complex_math (Torch::python::array<T,N>& array) {
      array.object()->def("acosh", &acosh<T,N>, "Inverse hyperbolic cosine, element-wise");
      array.object()->def("acos", &acos<T,N>, "Arc cosine, element-wise");
      array.object()->def("asin", &asin<T,N>, "Arc sine, element-wise");
      array.object()->def("atan", &atan<T,N>, "Arc tangent, element-wise");
      array.object()->def("atanh", &atanh<T,N>, "Inverse hyperbolic tangent, element-wise");
      array.object()->def("cbrt", &cbrt<T,N>, "self ** (1/3) (cubic root), element-wise");
      array.object()->def("exp", &exp<T,N>, "exponential element-wise");
      array.object()->def("expm1", &expm1<T,N>, "exp(x)-1, element-wise");
      array.object()->def("erf", &erf<T,N>, "Computes the error function: erf(x) = 2/sqrt(Pi) * integral(exp(-t^2), t=0..x) Note that for large values of the parameter, calculating 1.0-erf(x) can result in extreme loss of accuracy. Instead, use erfc(), element-wise");
      array.object()->def("erfc", &erfc<T,N>, "Computes the complementary error function erfc(x) = 1.0 - erf(x), element-wise.");
      array.object()->def("ilogb", &ilogb<T,N>, "Returns an integer which is equal to the unbiased exponent of the parameter, element-wise.");
      array.object()->def("rint", &rint<T,N>, "Rounds the parameter and returns a floating-point integer value. Whether rint() rounds up or down or to the nearest integer depends on the current floating-point rounding mode. If you haven't altered the rounding mode, rint() should be equivalent to nearest(). If rounding mode is set to round towards +INF, rint() is equivalent to ceil(). If the mode is round toward -INF, rint() is equivalent to floor(). If the mode is round toward zero, rint() is equivalent to trunc(). )");
      array.object()->def("j0", &j0<T,N>, "Bessel function of the first kind, order 0."); 
      array.object()->def("j1", &j1<T,N>, "Bessel function of the first kind, order 1."); 
      array.object()->def("lgamma", &lgamma<T,N>, "Natural logarithm of the gamma function. The gamma function Gamma(x) is defined as: Gamma(x) = integral(e^(-t) * t^(x-1), t=0..infinity)."); 
      array.object()->def("log1p", &log1p<T,N>, "Calculates log(1+x), where x is the parameter.");
      array.object()->def("y0", &y0<T,N>, "Bessel function of the second kind, order 0.");
      array.object()->def("y1", &y1<T,N>, "Bessel function of the second kind, order 1.");
    }

  template <typename T, int N>
    void bind_common_math (Torch::python::array<T,N>& array) {
      array.object()->def("fill", &fill<T,N>, (boost::python::arg("self"), boost::python::arg("value")), "Fills the array with the same value all over");
    }

  template <int N> void bind_bool_math (Torch::python::array<bool, N>& array) {
    bind_common_math(array);
  }

  template <typename T, int N> 
    void bind_int_math (Torch::python::array<T,N>& array) {
    bind_common_math(array);
    bind_non_bool_math(array);
    bind_non_complex_math(array);
  }

  template <typename T, int N> 
    void bind_float_math (Torch::python::array<T,N>& array) {
    bind_common_math(array);
    bind_non_bool_math(array);
    bind_non_complex_math(array);

    array.object()->def("ceil", &ceil<T,N>, "Ceiling function: smallest floating-point integer value not less than the argument."); 
    array.object()->def("floor", &floor<T,N>, "Floor function: largest floating-point integer value not greater than the argument.");

    //these are some free-standing operators
    boost::python::def("atan2", &atan2<T,N>, (boost::python::arg("x"), boost::python::arg("y")), "Inverse tangent of (y/x). The signs of both parameters are used to determine the quadrant of the return value, which is in the range [-pi, pi]. Works for complex<T>.");
    boost::python::def("radius", &radius<T,N>, (boost::python::arg("x"), boost::python::arg("y")), "Calculates the radius component in a cartesian-polar coordinate conversion");
  }

  template <typename T, int N> 
    void bind_complex_math (Torch::python::array<T,N>& array) {
    bind_common_math(array);
    bind_non_bool_math(array);

    array.object()->def("real", &blitz::real<typename T::value_type,N>, "Returns the real portion of the array (reference).");
    array.object()->def("imag", &blitz::imag<typename T::value_type,N>, "Returns the imag portion of the array (reference).");
    array.object()->def("arg", &arg<T,N>, "Argument of a complex number (atan2(Im,Re)).");
    array.object()->def("conj", conj<T,N>, "Conjugate of a complex number.");
  }

}}

#endif /* TORCH_PYTHON_CORE_ARRAY_MATH_H */

