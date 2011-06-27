/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::io::Array 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/preprocessor.hpp>

#include "io/Array.h"
#include "core/array_type.h"

using namespace boost::python;
namespace io = Torch::io;
namespace core = Torch::core;
namespace array = Torch::core::array;

typedef class_<io::Array, boost::shared_ptr<io::Array>, boost::shared_ptr<const io::Array> > PyClass;

template <typename T, int D> static boost::shared_ptr<io::Array> make_array(blitz::Array<T,D>& bz) {
  return boost::make_shared<io::Array>(io::detail::InlinedArrayImpl(bz));
}

template<typename T, int D> static void loop(PyClass& obj) {

  static const char* MAKE_ARRAY_DOC = "Creates a new io.Array from the given blitz::Array<T,D>.";
  static const char* ARRAY_CAST_DOC = "Adapts the size of each dimension of the passed blitz array to the ones of the underlying array and *copies* the data in it.";
  static const char* ARRAY_GET_DOC = "Adapts the size of each dimension of the passed blitz array to the ones of the underlying array and *refers* to the data in it. WARNING: Updating the content of the blitz array will update the content of the corresponding array in the dataset. Use this method with care!";

  boost::format f("__%s_%s_%d__");

  obj.def("__init__", make_constructor(&make_array<T,D>, default_call_policies(), arg("array")), MAKE_ARRAY_DOC); \
  obj.def((f % "cast" % array::stringize<T>() % D).str().c_str(), (blitz::Array<T,D> (io::Array::*)(void) const)&io::Array::cast<T,D>, (arg("self")), ARRAY_CAST_DOC); \
  obj.def((f % "get" % array::stringize<T>() % D).str().c_str(), (const blitz::Array<T,D> (io::Array::*)(void) const)&io::Array::get<T,D>, (arg("self")), ARRAY_GET_DOC); 

  implicitly_convertible<blitz::Array<T,D>, io::Array>();
}

static const char* get_filename(const io::Array& a) {
  return a.getFilename().c_str();
}

tuple get_shape(const io::Array& as) {
  size_t ndim = as.getNDim();
  const size_t* shape = as.getShape();
  switch (ndim) {
    case 1:
      return make_tuple(shape[0]);
    case 2:
      return make_tuple(shape[0], shape[1]);
    case 3:
      return make_tuple(shape[0], shape[1], shape[2]);
    case 4:
      return make_tuple(shape[0], shape[1], shape[2], shape[3]);
    default:
      break;
  }
  return make_tuple();
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(array_save_overloads, save, 1, 2) 

void bind_io_array() {

  enum_<array::ElementType>("ElementType")
    .value(array::stringize(array::t_unknown), array::t_unknown)
    .value(array::stringize(array::t_bool), array::t_bool)
    .value(array::stringize(array::t_int8), array::t_int8)
    .value(array::stringize(array::t_int16), array::t_int16)
    .value(array::stringize(array::t_int32), array::t_int32)
    .value(array::stringize(array::t_int64), array::t_int64)
    .value(array::stringize(array::t_uint8), array::t_uint8)
    .value(array::stringize(array::t_uint16), array::t_uint16)
    .value(array::stringize(array::t_uint32), array::t_uint32)
    .value(array::stringize(array::t_uint64), array::t_uint64)
    .value(array::stringize(array::t_float32), array::t_float32)
    .value(array::stringize(array::t_float64), array::t_float64)
    .value(array::stringize(array::t_float128), array::t_float128)
    .value(array::stringize(array::t_complex64), array::t_complex64)
    .value(array::stringize(array::t_complex128), array::t_complex128)
    .value(array::stringize(array::t_complex256), array::t_complex256)
    ;

  //base class declaration
  PyClass array("Array", "Dataset Arrays represent pointers to concrete data serialized on a io. You can load or refer to real blitz::Arrays using this type.", init<const std::string&, optional<const std::string&> >((arg("filename"),arg("codecname")=""), "Initializes a new array from an external file. An optional codec may be passed."));

  //attach generic methods
  array.def("save", &io::Array::save, array_save_overloads((arg("filename"), arg("codecname")=""), "Saves, renames or re-writes the array into a file. It will save if the array is loaded in memory. It will move if the codec used does not change by the filename does. It will re-write if the codec changes."));
  array.def("load", &io::Array::load);
  array.add_property("loaded", &io::Array::isLoaded);
  array.add_property("filename", &get_filename, "Accesses the filename containing the data for this array, if it was stored on a separate file. This string is empty otherwise.");
  array.add_property("codec", &io::Array::getCodec, "Accesses the codec that decodes and encodes this Array from/to files.");
  array.add_property("shape", &get_shape, "Accesses the shape of this array");
  array.add_property("elementType", &io::Array::getElementType, "Accesses the element type of this array");

  //loop for all supported dimensions using the boost preprocessor
# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  loop<bool,D>(array); \
  loop<int8_t,D>(array); \
  loop<int16_t,D>(array); \
  loop<int32_t,D>(array); \
  loop<int64_t,D>(array); \
  loop<uint8_t,D>(array); \
  loop<uint16_t,D>(array); \
  loop<uint32_t,D>(array); \
  loop<uint64_t,D>(array); \
  loop<float,D>(array); \
  loop<double,D>(array); \
  loop<double,D>(array); \
  loop<std::complex<float>,D>(array); \
  loop<std::complex<double>,D>(array); \
  loop<std::complex<double>,D>(array);
# include BOOST_PP_LOCAL_ITERATE()
}
