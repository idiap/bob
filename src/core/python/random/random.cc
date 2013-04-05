/**
 * @file core/python/random/random.cc
 * @date Mon Jul 11 18:31:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Bindings for random number generation.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdint.h>
#include <boost/random.hpp>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION >= 104700
#include <boost/random/discrete_distribution.hpp>
#include <bob/core/python/ndarray.h>
#endif

using namespace boost::python;

template <typename T>
static boost::shared_ptr<boost::mt19937> make_with_seed(T s) {
  return boost::make_shared<boost::mt19937>(s);
}

template <typename T>
static void set_seed(boost::mt19937& o, T s) {
  o.seed(s);
}

template <typename Distribution, typename Engine>
static typename Distribution::result_type __call__(Distribution& d, Engine& e) {
  return boost::variate_generator<Engine&,Distribution>(e,d)();
}

template <typename T, typename Engine>
static void uniform_int(const char* vartype) {
  typedef boost::uniform_int<T> D;

  boost::format name("uniform_%s");
  name % vartype;

  boost::format doc("The distribution class %s (boost::uniform_int<%s>) models a uniform random distribution. On each invocation, it returns a random integer value uniformly distributed in the set of integer numbers {min, min+1, min+2, ..., max}.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T, T> >((arg("min")=0, arg("max")=9), "Constructs a new object of this type, 'min' and 'max' are parameters of the distribution"))
    .add_property("min", &D::min)
    .add_property("max", &D::max)
    .def("reset", &D::reset, "This is a noop for this distribution, here only for consistency")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

template <typename T, typename Engine>
static void uniform_real(const char* vartype) {
  typedef boost::uniform_real<T> D;

  boost::format name("uniform_%s");
  name % vartype;

  boost::format doc("The distribution class %s (boost::uniform_real<%s>) models a random distribution. On each invocation, it returns a random floating-point value uniformly distributed in the range [min..max). The value is computed using std::numeric_limits<RealType>::digits random binary digits, i.e. the mantissa of the floating-point value is completely filled with random bits.\n\n.. note::\n   The current implementation is buggy, because it may not fill all of the mantissa with random bits.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T, T> >((arg("min")=0, arg("max")=1), "Constructs a new object of this type, 'min' and 'max' are parameters of the distribution. 'min' has to be <= 'max'."))
    .add_property("min", &D::min)
    .add_property("max", &D::max)
    .def("reset", &D::reset, "This is a noop for this distribution, here only for consistency")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

template <typename T, typename Engine>
static void normal_distribution(const char* vartype) {
  typedef boost::normal_distribution<T> D;

  boost::format name("normal_%s");
  name % vartype;

  boost::format doc("The distribution class %s (boost::normal_distribution<%s>) models a random distribution. Such a distribution produces random numbers 'x' distributed with the probability density function :math:`p(x) = \\frac{1}{\\sqrt{2\\pi\\sigma}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}`, where mean and sigma are the parameters of the distribution.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T, T> >((arg("mean")=0, arg("sigma")=1), "Constructs a new object of this type, 'mean' and 'sigma' are parameters of the distribution."))
    .add_property("mean", &D::mean)
    .add_property("sigma", &D::sigma)
    .def("reset", &D::reset, "resets the internal state")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

template <typename T, typename Engine>
static void lognormal_distribution(const char* vartype) {
  typedef boost::lognormal_distribution<T> D;

  boost::format name("lognormal_%s");
  name % vartype;

  boost::format doc("The distribution class %s (boost::lognormal_distribution<%s>) models a random distribution. Such a distribution produces random numbers 'x' distributed with the probability density function :math:`p(x) = \\frac{1}{x \\sigma_N \\sqrt{2\\pi}} e^{\\frac{-\\left(\\log(x)-\\mu_N\\right)^2}{2\\sigma_N^2}}`, for :math:`x > 0` and :math:`\\sigma_N = \\sqrt{\\log\\left(1 + \\frac{\\sigma^2}{\\mu^2}\\right)}`.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T, T> >((arg("mean")=1, arg("sigma")=1), "Constructs a new object of this type, 'mean' and 'sigma' are parameters of the distribution."))
    .add_property("mean", &D::mean)
    .add_property("sigma", &D::sigma)
    .def("reset", &D::reset, "resets the internal state")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

template <typename T, typename Engine>
static void gamma_distribution(const char* vartype) {
  typedef boost::gamma_distribution<T> D;

  boost::format name("gamma_%s");
  name % vartype;

  boost::format doc("The distribution class %s (boost::gamma_distribution<%s>) models a random distribution. The gamma distribution is a continuous distribution with a single parameter 'alpha'. It has :math:`p(x) = x^{\\alpha-1}\\frac{e^{-x}}{\\Gamma(\\alpha)}`.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T> >((arg("alpha")=1), "Constructs a new object of this type, 'alpha' is a parameter of the distribution."))
    .add_property("alpha", &D::alpha)
    .def("reset", &D::reset, "resets the internal state")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

template <typename T, typename I, typename Engine>
static void binomial_distribution(const char* vartype) {
  typedef boost::binomial_distribution<I, T> D;

  boost::format name("binomial_%s");
  name % vartype;

  boost::format doc("The distribution class %s (boost::binomial_distribution<%s>) models a random distribution. The binomial distribution is an integer valued distribution with two parameters, 't' and 'p'. The values of the distribution are within the range [0,t]. The probability that the distribution produces a value k is :math:`{{t}\\choose{k}}p^k(1-p)^{t-k}`.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def(init<optional<T, T> >((arg("t")=1, arg("p")=0.5), "Constructs a new object of this type, 't' and 'p' are parameters of the distribution. Requires :math:`t >=0` and :math:`0 <= p <= 1`."))
    .add_property("t", &D::t)
    .add_property("p", &D::p)
    .def("reset", &D::reset, "this is a noop for this distribution, but is here for consistence with other APIs")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}

#if BOOST_VERSION >= 104700
template <typename I>
static boost::shared_ptr<boost::random::discrete_distribution<I,double> > 
py_discrete_init_from_array(bob::python::const_ndarray p)
{
  const bob::core::array::typeinfo& info = p.type(); 
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot initialize with array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> p_ = p.bz<double,1>();
  const int len = p_.extent(0);
  std::vector<double> v_p(len);
  for(int i=0; i<len; ++i) v_p[i] = p_(i);
  return boost::make_shared<boost::random::discrete_distribution<I,double> >(v_p);
}

template <typename I>
static object py_discrete_get_prob(const boost::random::discrete_distribution<I,double>& D)
{
  std::vector<double> w = D.probabilities();
  const int len = w.size(); 
  bob::python::ndarray p(bob::core::array::t_float64, len);
  blitz::Array<double,1> p_ = p.bz<double,1>();
  for(int i=0; i<len; ++i) p_(i) = w[i];
  return p.self();
}

template <typename I, typename Engine>
static void discrete_distribution(const char* vartype) {
  typedef boost::random::discrete_distribution<I, double> D;

  boost::format name("discrete_%s");
  name % vartype;

  boost::format doc("The distribution class %s (boost::random::discrete_distribution<%s>) models a discrete distribution. The discrete distribution has a single parameter 'probabilities', which is a 1D numpy array containing the probability associated with each integer value in the range [0,t[, where t is the length of this vector. The values of the distribution are hence within the range [0,t[.");
  doc % name.str() % vartype;

  class_<D, boost::shared_ptr<D> >(name.str().c_str(), doc.str().c_str(), no_init)
    .def("__init__", make_constructor(&py_discrete_init_from_array<I>, default_call_policies(), (arg("probabilities"))), "Constructs a new object of this type; the 'probabilities' are the parameters of the discrete distribution.")
    .add_property("probabilities", &py_discrete_get_prob<I>)
    .def("reset", &D::reset, "this is a noop for this distribution, but is here for consistence with other APIs")
    .def("__call__", __call__<D,Engine>, (arg("self"), arg("rng")))
    ;
}
#endif

void bind_core_random () {
  class_<boost::mt19937, boost::shared_ptr<boost::mt19937> >("mt19937", "A Random Number Generator (RNG) based on the work 'Mersenne Twister: A 623-dimensionally equidistributed uniform pseudo-random number generator, Makoto Matsumoto and Takuji Nishimura, ACM Transactions on Modeling and Computer Simulation: Special Issue on Uniform Random Number Generation, Vol. 8, No. 1, January 1998, pp. 3-30'", init<>("Default constructor"))
    .def("__init__", make_constructor(&make_with_seed<int16_t>, default_call_policies(), (arg("seed"))), "Builds a new generator with a specific seed")
    .def("__init__", make_constructor(&make_with_seed<int32_t>, default_call_policies(), (arg("seed"))), "Builds a new generator with a specific seed")
    .def("__init__", make_constructor(&make_with_seed<int64_t>, default_call_policies(), (arg("seed"))), "Builds a new generator with a specific seed")
    .def("__init__", make_constructor(&make_with_seed<double>, default_call_policies(), (arg("seed"))), "Builds a new generator with a specific seed")
    .def("seed", &set_seed<double>, "Sets my internal seed")
    .def(self == self)
    .def(self != self)
    ;

  uniform_int<int8_t, boost::mt19937>("int8");
  uniform_int<int16_t, boost::mt19937>("int16");
  uniform_int<int32_t, boost::mt19937>("int32");
  uniform_int<int64_t, boost::mt19937>("int64");
  uniform_int<uint8_t, boost::mt19937>("uint16");
  uniform_int<uint16_t, boost::mt19937>("uint32");
  uniform_int<uint32_t, boost::mt19937>("uint64");
  uniform_int<uint64_t, boost::mt19937>("uint64");
  uniform_real<float, boost::mt19937>("float32");
  uniform_real<double, boost::mt19937>("float64");
  normal_distribution<float, boost::mt19937>("float32");
  normal_distribution<double, boost::mt19937>("float64");
  lognormal_distribution<float, boost::mt19937>("float32");
  lognormal_distribution<double, boost::mt19937>("float64");
  gamma_distribution<float, boost::mt19937>("float32");
  gamma_distribution<double, boost::mt19937>("float64");
  binomial_distribution<float, int64_t, boost::mt19937>("float32");
  binomial_distribution<double, int64_t, boost::mt19937>("float64");

#if BOOST_VERSION >= 104700
  discrete_distribution<int8_t, boost::mt19937>("int8");
  discrete_distribution<int16_t, boost::mt19937>("int16");
  discrete_distribution<int32_t, boost::mt19937>("int32");
  discrete_distribution<int64_t, boost::mt19937>("int64");
  discrete_distribution<uint8_t, boost::mt19937>("uint8");
  discrete_distribution<uint16_t, boost::mt19937>("uint16");
  discrete_distribution<uint32_t, boost::mt19937>("uint32");
  discrete_distribution<uint64_t, boost::mt19937>("uint64");
#endif
}
