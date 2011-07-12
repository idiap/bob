/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 11 Jul 2011 16:39:16 CEST
 *
 * @brief Bindings for random number generation.
 */

#include <stdint.h>
#include <boost/random.hpp>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

using namespace boost::python;

static boost::shared_ptr<boost::mt19937> make_with_seed(size_t s) {
  return boost::make_shared<boost::mt19937>(s);
}

static void set_seed(boost::mt19937& o, size_t s) {
  o.seed(s);
}

template <typename T, typename Distribution> struct binder {

  typedef boost::variate_generator<boost::mt19937&, Distribution > dtype;
  typedef typename dtype::result_type result_type;

  boost::shared_ptr<dtype> make_dtype_default() {
    boost::mt19937 rng;
    Distribution d;
    return boost::shared_ptr<dtype>(new dtype(rng, d));
  }

  boost::shared_ptr<dtype> make_dtype(T arg1) {
    boost::mt19937 rng;
    Distribution d(arg1);
    return boost::shared_ptr<dtype>(new dtype(rng, d));
  }

  void set_seed(dtype& gen, size_t s) {
    gen.engine().seed(s);
    gen.distribution().reset();
  }

  binder(const char* name, const char* par1) {
    class_<dtype, boost::shared_ptr<dtype> >(name, "A variate generator", no_init)
      .def("__init__", make_constructor(&binder<T,Distribution>::make_dtype_default), "Starts a new generator with default parameters. For information about defaults, consult the C++ boost::random documentation available on the web.")
      .def("__init__", make_constructor(&binder<T,Distribution>::make_dtype, default_call_policies(), (arg(par1))), "Starts a new generator with parameters. For detailed information and range inclusion, consult the C++ boost::random documentation available on the web.")
      .def("seed", &binder<T,Distribution>::set_seed, (arg("self"), arg("seed")), "Sets the internal seed of my own RNG and reset my distribution")
      .def("__call__", (result_type (dtype::*)())&dtype::operator(), (arg("self")), "Draws a new number")
      ;
  }

};

template <typename T, typename Distribution> struct binder2 {

  typedef boost::variate_generator<boost::mt19937&, Distribution > dtype;
  typedef typename dtype::result_type result_type;

  boost::shared_ptr<dtype> make_dtype_default() {
    boost::mt19937 rng;
    Distribution d;
    return boost::shared_ptr<dtype>(new dtype(rng, d));
  }

  boost::shared_ptr<dtype> make_dtype(T arg1, T arg2) {
    boost::mt19937 rng;
    Distribution d(arg1, arg2);
    return boost::shared_ptr<dtype>(new dtype(rng, d));
  }

  void set_seed(dtype& gen, size_t s) {
    gen.engine().seed(s);
    gen.distribution().reset();
  }

  binder2(const char* name, const char* par1, const char* par2) {
    class_<dtype, boost::shared_ptr<dtype> >(name, "A variate generator", no_init)
      .def("__init__", make_constructor(&binder2<T,Distribution>::make_dtype_default), "Starts a new generator with default parameters. For information about defaults, consult the C++ boost::random documentation available on the web.")
      .def("__init__", make_constructor(&binder2<T,Distribution>::make_dtype, default_call_policies(), (arg(par1), arg(par2))), "Starts a new generator with parameters. Note that, for range style parameters (in uniform distributions for example), integer ranges include border values while for real-valued distributions, the lower end is included but not the upper end [to,from). For detailed information and range inclusion, consult the C++ boost::random documentation available on the web.")
      .def("seed", &binder2<T,Distribution>::set_seed, (arg("self"), arg("seed")), "Sets the internal seed of my own RNG and reset my distribution")
      .def("__call__", (result_type (dtype::*)())&dtype::operator(), (arg("self")), "Draws a new number")
      ;
  }

};

void bind_core_random () {
  class_<boost::mt19937, boost::shared_ptr<boost::mt19937> >("mt19937", "A Random Number Generator (RNG) based on the work 'Mersenne Twister: A 623-dimensionally equidistributed uniform pseudo-random number generator, Makoto Matsumoto and Takuji Nishimura, ACM Transactions on Modeling and Computer Simulation: Special Issue on Uniform Random Number Generation, Vol. 8, No. 1, January 1998, pp. 3-30'", init<>("Default constructor"))
    .def("__init__", make_constructor(&make_with_seed, default_call_policies(), (arg("seed"))), "Builds a new generator with a specific seed")
    .add_property("seed", &set_seed, "Sets the seed of the random number generator")
    ;

  //single argument methods
  binder<float, boost::gamma_distribution<float> >("gamma_float32", "alpha");
  binder<double, boost::gamma_distribution<double> >("gamma_float64", "alpha");

  //two-argument methods
  binder2<int8_t, boost::uniform_smallint<int8_t> >("uniform_int8", "from", "to");
  binder2<int16_t, boost::uniform_int<int16_t> >("uniform_int16", "from", "to");
  binder2<int32_t, boost::uniform_int<int32_t> >("uniform_int32", "from", "to");
  binder2<int64_t, boost::uniform_int<int64_t> >("uniform_int64", "from", "to");
  binder2<uint8_t, boost::uniform_smallint<uint8_t> >("uniform_uint8", "from", "to");
  binder2<uint16_t, boost::uniform_int<uint16_t> >("uniform_uint16", "from", "to");
  binder2<uint32_t, boost::uniform_int<uint32_t> >("uniform_uint32", "from", "to");
  binder2<uint64_t, boost::uniform_int<uint64_t> >("uniform_uint64", "from", "to");
  binder2<float, boost::uniform_real<float> >("uniform_float32", "from", "to");
  binder2<double, boost::uniform_real<double> >("uniform_float64", "from", "to");
  binder2<float, boost::binomial_distribution<float> >("binomial_float32", "from", "to");
  binder2<double, boost::binomial_distribution<double> >("binomial_float64", "from", "to");
  binder2<float, boost::normal_distribution<float> >("normal_float32", "mean", "sigma");
  binder2<double, boost::normal_distribution<double> >("normal_float64", "mean", "sigma");
  binder2<float, boost::lognormal_distribution<float> >("lognormal_float32", "mean", "sigma");
  binder2<double, boost::lognormal_distribution<double> >("lognormal_float64", "mean", "sigma");
}
