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

tp::array<bool, 1> tp::bool_1("bool");
tp::array<bool, 2> tp::bool_2("bool");
tp::array<bool, 3> tp::bool_3("bool");
tp::array<bool, 4> tp::bool_4("bool");

tp::array<int8_t, 1> tp::int8_1("int8");
tp::array<int8_t, 2> tp::int8_2("int8");
tp::array<int8_t, 3> tp::int8_3("int8");
tp::array<int8_t, 4> tp::int8_4("int8");

tp::array<int16_t, 1> tp::int16_1("int16");
tp::array<int16_t, 2> tp::int16_2("int16");
tp::array<int16_t, 3> tp::int16_3("int16");
tp::array<int16_t, 4> tp::int16_4("int16");

tp::array<int32_t, 1> tp::int32_1("int32");
tp::array<int32_t, 2> tp::int32_2("int32");
tp::array<int32_t, 3> tp::int32_3("int32");
tp::array<int32_t, 4> tp::int32_4("int32");

tp::array<int64_t, 1> tp::int64_1("int64");
tp::array<int64_t, 2> tp::int64_2("int64");
tp::array<int64_t, 3> tp::int64_3("int64");
tp::array<int64_t, 4> tp::int64_4("int64");

tp::array<uint8_t, 1> tp::uint8_1("uint8");
tp::array<uint8_t, 2> tp::uint8_2("uint8");
tp::array<uint8_t, 3> tp::uint8_3("uint8");
tp::array<uint8_t, 4> tp::uint8_4("uint8");

tp::array<uint16_t, 1> tp::uint16_1("uint16");
tp::array<uint16_t, 2> tp::uint16_2("uint16");
tp::array<uint16_t, 3> tp::uint16_3("uint16");
tp::array<uint16_t, 4> tp::uint16_4("uint16");

tp::array<uint32_t, 1> tp::uint32_1("uint32");
tp::array<uint32_t, 2> tp::uint32_2("uint32");
tp::array<uint32_t, 3> tp::uint32_3("uint32");
tp::array<uint32_t, 4> tp::uint32_4("uint32");

tp::array<uint64_t, 1> tp::uint64_1("uint64");
tp::array<uint64_t, 2> tp::uint64_2("uint64");
tp::array<uint64_t, 3> tp::uint64_3("uint64");
tp::array<uint64_t, 4> tp::uint64_4("uint64");

tp::array<float, 1> tp::float32_1("float32");
tp::array<float, 2> tp::float32_2("float32");
tp::array<float, 3> tp::float32_3("float32");
tp::array<float, 4> tp::float32_4("float32");

tp::array<double, 1> tp::float64_1("float64");
tp::array<double, 2> tp::float64_2("float64");
tp::array<double, 3> tp::float64_3("float64");
tp::array<double, 4> tp::float64_4("float64");

tp::array<long double, 1> tp::float128_1("float128");
tp::array<long double, 2> tp::float128_2("float128");
tp::array<long double, 3> tp::float128_3("float128");
tp::array<long double, 4> tp::float128_4("float128");

tp::array<std::complex<float>, 1> tp::complex64_1("complex64");
tp::array<std::complex<float>, 2> tp::complex64_2("complex64");
tp::array<std::complex<float>, 3> tp::complex64_3("complex64");
tp::array<std::complex<float>, 4> tp::complex64_4("complex64");

tp::array<std::complex<double>, 1> tp::complex128_1("complex128");
tp::array<std::complex<double>, 2> tp::complex128_2("complex128");
tp::array<std::complex<double>, 3> tp::complex128_3("complex128");
tp::array<std::complex<double>, 4> tp::complex128_4("complex128");

tp::array<std::complex<long double>, 1> tp::complex256_1("complex256");
tp::array<std::complex<long double>, 2> tp::complex256_2("complex256");
tp::array<std::complex<long double>, 3> tp::complex256_3("complex256");
tp::array<std::complex<long double>, 4> tp::complex256_4("complex256");

void bind_array_base () {
  tp::bool_1.bind(); register_class(tp::bool_1);  
  tp::bool_2.bind(); register_class(tp::bool_2);
  tp::bool_3.bind(); register_class(tp::bool_3);
  tp::bool_4.bind(); register_class(tp::bool_4);
  tp::int8_1.bind(); register_class(tp::int8_1);
  tp::int8_2.bind(); register_class(tp::int8_2);
  tp::int8_3.bind(); register_class(tp::int8_3);
  tp::int8_4.bind(); register_class(tp::int8_4);
  tp::int16_1.bind(); register_class(tp::int16_1);
  tp::int16_2.bind(); register_class(tp::int16_2);
  tp::int16_3.bind(); register_class(tp::int16_3);
  tp::int16_4.bind(); register_class(tp::int16_4);
  tp::int32_1.bind(); register_class(tp::int32_1);
  tp::int32_2.bind(); register_class(tp::int32_2);
  tp::int32_3.bind(); register_class(tp::int32_3);
  tp::int32_4.bind(); register_class(tp::int32_4);
  tp::int64_1.bind(); register_class(tp::int64_1);
  tp::int64_2.bind(); register_class(tp::int64_2);
  tp::int64_3.bind(); register_class(tp::int64_3);
  tp::int64_4.bind(); register_class(tp::int64_4);
  tp::uint8_1.bind(); register_class(tp::uint8_1);
  tp::uint8_2.bind(); register_class(tp::uint8_2);
  tp::uint8_3.bind(); register_class(tp::uint8_3);
  tp::uint8_4.bind(); register_class(tp::uint8_4);
  tp::uint16_1.bind(); register_class(tp::uint16_1);
  tp::uint16_2.bind(); register_class(tp::uint16_2);
  tp::uint16_3.bind(); register_class(tp::uint16_3);
  tp::uint16_4.bind(); register_class(tp::uint16_4);
  tp::uint32_1.bind(); register_class(tp::uint32_1);
  tp::uint32_2.bind(); register_class(tp::uint32_2);
  tp::uint32_3.bind(); register_class(tp::uint32_3);
  tp::uint32_4.bind(); register_class(tp::uint32_4);
  tp::uint64_1.bind(); register_class(tp::uint64_1);
  tp::uint64_2.bind(); register_class(tp::uint64_2);
  tp::uint64_3.bind(); register_class(tp::uint64_3);
  tp::uint64_4.bind(); register_class(tp::uint64_4);
  tp::float32_1.bind(); register_class(tp::float32_1);
  tp::float32_2.bind(); register_class(tp::float32_2);
  tp::float32_3.bind(); register_class(tp::float32_3);
  tp::float32_4.bind(); register_class(tp::float32_4);
  tp::float64_1.bind(); register_class(tp::float64_1);
  tp::float64_2.bind(); register_class(tp::float64_2);
  tp::float64_3.bind(); register_class(tp::float64_3);
  tp::float64_4.bind(); register_class(tp::float64_4);
  tp::float128_1.bind(); register_class(tp::float128_1);
  tp::float128_2.bind(); register_class(tp::float128_2);
  tp::float128_3.bind(); register_class(tp::float128_3);
  tp::float128_4.bind(); register_class(tp::float128_4);
  tp::complex64_1.bind(); register_class(tp::complex64_1);
  tp::complex64_2.bind(); register_class(tp::complex64_2);
  tp::complex64_3.bind(); register_class(tp::complex64_3);
  tp::complex64_4.bind(); register_class(tp::complex64_4);
  tp::complex128_1.bind(); register_class(tp::complex128_1);
  tp::complex128_2.bind(); register_class(tp::complex128_2);
  tp::complex128_3.bind(); register_class(tp::complex128_3);
  tp::complex128_4.bind(); register_class(tp::complex128_4);
  tp::complex256_1.bind(); register_class(tp::complex256_1);
  tp::complex256_2.bind(); register_class(tp::complex256_2);
  tp::complex256_3.bind(); register_class(tp::complex256_3);
  tp::complex256_4.bind(); register_class(tp::complex256_4);
}
