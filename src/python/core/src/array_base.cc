/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  9 Mar 17:04:47 2011 
 *
 * @brief Declares our basic support blitz::Array<> types in python
 */

#include "core/python/array_base.h"
#include <map>

namespace tp = Torch::python;
namespace bp = boost::python;
namespace tca = Torch::core::array;

static std::map<std::pair<tca::ElementType, int>, bp::object> classes;

bp::object tp::array_class(tca::ElementType eltype, int rank) {
  std::map<std::pair<tca::ElementType, int>, bp::object>::const_iterator it = classes.find(std::make_pair(eltype, rank));
  if (it == classes.end()) {
    boost::format msg("No support for blitz::Array<%s,%d> in python");
    msg % tca::stringize(eltype) % rank;
    PyErr_SetString(PyExc_TypeError, msg.str().c_str());
    boost::python::throw_error_already_set();
  }
  return it->second;
}

template <typename T, int N>
static void register_class (tp::array<T,N>& a) {
  classes[std::make_pair(tca::getElementType<T>(), N)] = *a.object();
}

//global variable instantiation
namespace Torch { namespace python {
#  define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
#  define BOOST_PP_LOCAL_MACRO(D) \
   array<bool, D> BOOST_PP_CAT(bool_,D);\
   array<int8_t, D> BOOST_PP_CAT(int8_,D);\
   array<int16_t, D> BOOST_PP_CAT(int16_,D);\
   array<int32_t, D> BOOST_PP_CAT(int32_,D);\
   array<int64_t, D> BOOST_PP_CAT(int64_,D);\
   array<uint8_t, D> BOOST_PP_CAT(uint8_,D);\
   array<uint16_t, D> BOOST_PP_CAT(uint16_,D);\
   array<uint32_t, D> BOOST_PP_CAT(uint32_,D);\
   array<uint64_t, D> BOOST_PP_CAT(uint64_,D);\
   array<float, D> BOOST_PP_CAT(float32_,D);\
   array<double, D> BOOST_PP_CAT(float64_,D);\
   array<long double, D> BOOST_PP_CAT(float128_,D);\
   array<std::complex<float>, D> BOOST_PP_CAT(complex64_,D);\
   array<std::complex<double>, D> BOOST_PP_CAT(complex128_,D);\
   array<std::complex<long double>, D> BOOST_PP_CAT(complex256_,D);
#  include BOOST_PP_LOCAL_ITERATE()
}}

void bind_array_base () {

  using namespace Torch::python;

# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  BOOST_PP_CAT(bool_,D).bind(); register_class(BOOST_PP_CAT(bool_,D));\
  BOOST_PP_CAT(int8_,D).bind(); register_class(BOOST_PP_CAT(int8_,D));\
  BOOST_PP_CAT(int16_,D).bind(); register_class(BOOST_PP_CAT(int16_,D));\
  BOOST_PP_CAT(int32_,D).bind(); register_class(BOOST_PP_CAT(int32_,D));\
  BOOST_PP_CAT(int64_,D).bind(); register_class(BOOST_PP_CAT(int64_,D));\
  BOOST_PP_CAT(uint8_,D).bind(); register_class(BOOST_PP_CAT(uint8_,D));\
  BOOST_PP_CAT(uint16_,D).bind(); register_class(BOOST_PP_CAT(uint16_,D));\
  BOOST_PP_CAT(uint32_,D).bind(); register_class(BOOST_PP_CAT(uint32_,D));\
  BOOST_PP_CAT(uint64_,D).bind(); register_class(BOOST_PP_CAT(uint64_,D));\
  BOOST_PP_CAT(float32_,D).bind(); register_class(BOOST_PP_CAT(float32_,D));\
  BOOST_PP_CAT(float64_,D).bind(); register_class(BOOST_PP_CAT(float64_,D));\
  BOOST_PP_CAT(float128_,D).bind(); register_class(BOOST_PP_CAT(float128_,D));\
  BOOST_PP_CAT(complex64_,D).bind(); register_class(BOOST_PP_CAT(complex64_,D));\
  BOOST_PP_CAT(complex128_,D).bind(); register_class(BOOST_PP_CAT(complex128_,D));\
  BOOST_PP_CAT(complex256_,D).bind(); register_class(BOOST_PP_CAT(complex256_,D));
# include BOOST_PP_LOCAL_ITERATE()
}
