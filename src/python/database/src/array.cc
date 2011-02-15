/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::database::Array 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "database/Array.h"

using namespace boost::python;
namespace db = Torch::database;
namespace core = Torch::core;
namespace array = Torch::core::array;

template <typename T,int D> static boost::shared_ptr<db::Array> make_array(blitz::Array<T,D>& bz) {
  return boost::make_shared<db::Array>(db::detail::InlinedArrayImpl(bz));
}

static const char* MAKE_ARRAY_DOC = "Creates a new database.Array from the given blitz::Array<T,D>.";
#define MAKE_ARRAY_DEF(T,N,D) .def("__init__", make_constructor(&make_array<T,D>, default_call_policies(), arg("array")), MAKE_ARRAY_DOC)

static const char* ARRAY_CAST_DOC = "Adapts the size of each dimension of the passed blitz array to the ones of the underlying array and *copies* the data in it.";
#define ARRAY_CAST_DEF(T,N,D) .def(BOOST_PP_STRINGIZE(cast_ ## N ## _ ## D), (blitz::Array<T,D> (db::Array::*)(void) const)&db::Array::cast<T,D>, (arg("self")), ARRAY_CAST_DOC)

static const char* ARRAY_GET_DOC = "Adapts the size of each dimension of the passed blitz array to the ones of the underlying array and *refers* to the data in it. WARNING: Updating the content of the blitz array will update the content of the corresponding array in the dataset. Use this method with care!";
#define ARRAY_GET_DEF(T,N,D) .def(BOOST_PP_STRINGIZE(get_ ## N ## _ ## D), (const blitz::Array<T,D> (db::Array::*)(void) const)&db::Array::get<T,D>, (arg("self")), ARRAY_GET_DOC)

#define ARRAY_ALL_DEFS(T,N,D)\
  MAKE_ARRAY_DEF(T,N,D) \
  ARRAY_CAST_DEF(T,N,D) \
  ARRAY_GET_DEF(T,N,D)

static const char* get_filename(const db::Array& a) {
  return a.getFilename().c_str();
}

tuple get_shape(const db::Array& as) {
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

void bind_database_array() {
  enum_<array::ElementType>("ElementType")
    .value("unknown", array::t_unknown)
    .value("bool", array::t_bool)
    .value("int8", array::t_int8)
    .value("int16", array::t_int16)
    .value("int32", array::t_int32)
    .value("int64", array::t_int64)
    .value("uint8", array::t_uint8)
    .value("uint16", array::t_uint16)
    .value("uint32", array::t_uint32)
    .value("uint64", array::t_uint64)
    .value("float32", array::t_float32)
    .value("float64", array::t_float64)
    .value("float128", array::t_float128)
    .value("complex64", array::t_complex64)
    .value("complex128", array::t_complex128)
    .value("complex256", array::t_complex256)
    ;

  class_<db::Array, boost::shared_ptr<db::Array> >("Array", "Dataset Arrays represent pointers to concrete data serialized on a database. You can load or refer to real blitz::Arrays using this type.", init<const std::string&, optional<const std::string&> >((arg("filename"),arg("codecname")=""), "Initializes a new array from an external file. An optional codec may be passed."))
    .def("save", &db::Array::save, array_save_overloads((arg("filename"), arg("codecname")=""), "Saves, renames or re-writes the array into a file. It will save if the array is loaded in memory. It will move if the codec used does not change by the filename does. It will re-write if the codec changes."))
    .def("load", &db::Array::load)
    .add_property("loaded", &db::Array::isLoaded)
    .add_property("filename", &get_filename, "Accesses the filename containing the data for this array, if it was stored on a separate file. This string is empty otherwise.")
    .add_property("codec", &db::Array::getCodec, "Accesses the codec that decodes and encodes this Array from/to files.")
    .add_property("shape", &get_shape, "Accesses the shape of this array")
    .add_property("elementType", &db::Array::getElementType, "Accesses the element type of this array")
    .add_property("id", &db::Array::getId, &db::Array::setId, "Accesses and sets the id of this array, if any (if id == 0, it means unassigned, let for my parent to decide - that is normally a good strategy!)")
    ARRAY_ALL_DEFS(bool, bool, 1)
    ARRAY_ALL_DEFS(int8_t, int8, 1)
    ARRAY_ALL_DEFS(int16_t, int16, 1)
    ARRAY_ALL_DEFS(int32_t, int32, 1)
    ARRAY_ALL_DEFS(int64_t, int64, 1)
    ARRAY_ALL_DEFS(uint8_t, uint8, 1)
    ARRAY_ALL_DEFS(uint16_t, uint16, 1)
    ARRAY_ALL_DEFS(uint32_t, uint32, 1)
    ARRAY_ALL_DEFS(uint64_t, uint64, 1)
    ARRAY_ALL_DEFS(float, float32, 1)
    ARRAY_ALL_DEFS(double, float64, 1)
    ARRAY_ALL_DEFS(double, float128, 1)
    ARRAY_ALL_DEFS(std::complex<float>, complex64, 1)
    ARRAY_ALL_DEFS(std::complex<double>, complex128, 1)
    ARRAY_ALL_DEFS(std::complex<double>, complex256, 1)
    ARRAY_ALL_DEFS(bool, bool, 2)
    ARRAY_ALL_DEFS(int8_t, int8, 2)
    ARRAY_ALL_DEFS(int16_t, int16, 2)
    ARRAY_ALL_DEFS(int32_t, int32, 2)
    ARRAY_ALL_DEFS(int64_t, int64, 2)
    ARRAY_ALL_DEFS(uint8_t, uint8, 2)
    ARRAY_ALL_DEFS(uint16_t, uint16, 2)
    ARRAY_ALL_DEFS(uint32_t, uint32, 2)
    ARRAY_ALL_DEFS(uint64_t, uint64, 2)
    ARRAY_ALL_DEFS(float, float32, 2)
    ARRAY_ALL_DEFS(double, float64, 2)
    ARRAY_ALL_DEFS(double, float128, 2)
    ARRAY_ALL_DEFS(std::complex<float>, complex64, 2)
    ARRAY_ALL_DEFS(std::complex<double>, complex128, 2)
    ARRAY_ALL_DEFS(std::complex<double>, complex256, 2)
    ARRAY_ALL_DEFS(bool, bool, 3)
    ARRAY_ALL_DEFS(int8_t, int8, 3)
    ARRAY_ALL_DEFS(int16_t, int16, 3)
    ARRAY_ALL_DEFS(int32_t, int32, 3)
    ARRAY_ALL_DEFS(int64_t, int64, 3)
    ARRAY_ALL_DEFS(uint8_t, uint8, 3)
    ARRAY_ALL_DEFS(uint16_t, uint16, 3)
    ARRAY_ALL_DEFS(uint32_t, uint32, 3)
    ARRAY_ALL_DEFS(uint64_t, uint64, 3)
    ARRAY_ALL_DEFS(float, float32, 3)
    ARRAY_ALL_DEFS(double, float64, 3)
    ARRAY_ALL_DEFS(double, float128, 3)
    ARRAY_ALL_DEFS(std::complex<float>, complex64, 3)
    ARRAY_ALL_DEFS(std::complex<double>, complex128, 3)
    ARRAY_ALL_DEFS(std::complex<double>, complex256, 3)
    ARRAY_ALL_DEFS(bool, bool, 4)
    ARRAY_ALL_DEFS(int8_t, int8, 4)
    ARRAY_ALL_DEFS(int16_t, int16, 4)
    ARRAY_ALL_DEFS(int32_t, int32, 4)
    ARRAY_ALL_DEFS(int64_t, int64, 4)
    ARRAY_ALL_DEFS(uint8_t, uint8, 4)
    ARRAY_ALL_DEFS(uint16_t, uint16, 4)
    ARRAY_ALL_DEFS(uint32_t, uint32, 4)
    ARRAY_ALL_DEFS(uint64_t, uint64, 4)
    ARRAY_ALL_DEFS(float, float32, 4)
    ARRAY_ALL_DEFS(double, float64, 4)
    ARRAY_ALL_DEFS(double, float128, 4)
    ARRAY_ALL_DEFS(std::complex<float>, complex64, 4)
    ARRAY_ALL_DEFS(std::complex<double>, complex128, 4)
    ARRAY_ALL_DEFS(std::complex<double>, complex256, 4)
    ;
}
