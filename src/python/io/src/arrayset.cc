/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::io::Arrayset
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <blitz/array.h>

#include "io/Arrayset.h"
#include "core/array_assert.h"
#include "core/python/vector.h"
#include "core/python/exception.h"

using namespace boost::python;
namespace io = Torch::io;
namespace core = Torch::core;
namespace array = Torch::core::array;
namespace tp = Torch::python;

tuple get_shape(const io::Arrayset& as) {
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

static const char* get_filename(io::Arrayset& as) {
  return as.getFilename().c_str();
}

template <typename T>
static void pythonic_set (io::Arrayset& as, size_t id, T obj) {
  if (id < as.size()) PYTHON_ERROR(IndexError, "out of range");
  as.set(id, obj);
}

template<typename T>
static boost::shared_ptr<io::Arrayset> make_from_array_iterable(T iter) {
  boost::shared_ptr<io::Arrayset> retval = boost::make_shared<io::Arrayset>();
  stl_input_iterator<io::Array> end;
  for (stl_input_iterator<io::Array> it(iter); it != end; ++it) {
    retval->add(*it); //calls extract<io::Array>(iter[i])
  }
  return retval;
}


// Partial list of N ranges
#define range_1 blitz::Range::all()
#define range_2 range_1, blitz::Range::all()
#define range_3 range_2, blitz::Range::all()

// get_arrayX_Y gets the parameters to slice an array of Y dimensions
// over the X dimension
#define get_array0_2 i, range_1
#define get_array1_2 range_1, i

#define get_array0_3 i, range_2
#define get_array1_3 range_1, i, range_1
#define get_array2_3 range_2, i

#define get_array0_4 i, range_3
#define get_array1_4 range_1, i, range_2
#define get_array2_4 range_2, i, range_1
#define get_array3_4 range_3, i

// Declare a function add_array_dim_D (add an array to an arrayset)
#define add_array_(dim, D)\
template<typename T>\
static void add_array_##dim##_##D(io::Arrayset& arrayset, const blitz::Array<T, D>& array) {\
  for (int i = 0; i < array.extent(0); i++) {\
    blitz::Array<T, D-1> tmp = array(get_array##dim##_##D);\
    arrayset.add(tmp);\
  }\
}

// Templated add_array function, specialized below
template<int dim, int D>
static void add_array(io::Arrayset& arrayset, io::Array& array) {
  throw io::DimensionError(0, 0);
}

#define CASE_TYPE(ctype, type, dim, D) \
case ctype:\
  add_array_##dim##_##D(arrayset, array.get<type, D>());\
  break;


// Define a specialized add_array function
#define add_array(dim, D)\
add_array_(dim, D)\
template<>\
inline void add_array<dim, D>(io::Arrayset& arrayset, io::Array& array) {\
  switch (array.getElementType()) {\
    CASE_TYPE(array::t_bool,  bool,     dim, D)\
    \
    CASE_TYPE(array::t_int8,  int8_t,   dim, D)\
    CASE_TYPE(array::t_int16, int16_t,  dim, D)\
    CASE_TYPE(array::t_int32, int32_t,  dim, D)\
    CASE_TYPE(array::t_int64, int64_t,  dim, D)\
    \
    CASE_TYPE(array::t_uint8,  uint8_t,  dim, D)\
    CASE_TYPE(array::t_uint16, uint16_t, dim, D)\
    CASE_TYPE(array::t_uint32, uint32_t, dim, D)\
    CASE_TYPE(array::t_uint64, uint64_t, dim, D)\
    \
    CASE_TYPE(array::t_complex64,  std::complex<float>,       dim, D)\
    CASE_TYPE(array::t_complex128, std::complex<double>,      dim, D)\
    CASE_TYPE(array::t_complex256, std::complex<long double>, dim, D)\
    \
    default:\
      throw io::TypeError(array.getElementType(), array::getElementType<double>());\
  }\
}

// Declare all possible specialized add_array functions
add_array(0, 2)
add_array(1, 2)

add_array(0, 3)
add_array(1, 3)
add_array(2, 3)

add_array(0, 4)
add_array(1, 4)
add_array(2, 4)
add_array(3, 4)

// Switch case for the D
#define CASE_DIM(dim, D) \
case dim:\
  switch (D) {\
    case 2:  add_array<dim, 2>(arrayset, array); break;\
    case 3:  add_array<dim, 3>(arrayset, array); break;\
    case 4:  add_array<dim, 4>(arrayset, array); break;\
    default:\
      throw io::DimensionError(D, 4);\
  }\
  break;

void append_array(io::Arrayset& arrayset, io::Array& array, int dim) {
  switch (dim) {
    CASE_DIM(0, array.getNDim())
    CASE_DIM(1, array.getNDim())
    CASE_DIM(2, array.getNDim())
    CASE_DIM(3, array.getNDim())
    default:
      throw io::DimensionError(dim, 3);
  }
}


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(arrayset_save_overloads, save, 1, 2)

void bind_io_arrayset() {
  class_<io::Arrayset, boost::shared_ptr<io::Arrayset> >("Arrayset", "Dataset Arraysets represent lists of Arrays that share the same element type and dimension properties and are grouped together by the DB designer.", init<const std::string&, optional<const std::string&> >((arg("filename"),arg("codecname")=""), "Initializes a new arrayset from an external file. An optional codec may be passed."))
    .def("__init__", make_constructor(make_from_array_iterable<tuple>), "Creates a new Arrayset from a python tuple containing Arrays.")
    .def("__init__", make_constructor(make_from_array_iterable<list>), "Creates a new Arrayset from a python list containing Arrays.")
    .def(init<>("Creates a new empty arraset with an inlined representation."))
    .add_property("shape", &get_shape, "The shape of each array in this set is determined by this variable.")
    .add_property("loaded", &io::Arrayset::isLoaded, "This variable determines if the arrayset is loaded in memory. This may be false in the case the arrayset is completely stored in an external file.")
    .add_property("filename", &get_filename)
    .add_property("elementType", &io::Arrayset::getElementType, "This property indicates the type of element used for each array in the current set.")
    .def("save", &io::Arrayset::save, arrayset_save_overloads((arg("filename"), arg("codecname")=""), "Saves, renames or re-writes the arrayset into a file. It will save if the arrayset is loaded in memory. It will move if the codec used does not change by the filename does. It will re-write if the codec changes."))
    .def("load", &io::Arrayset::load)
    .def("extend", &append_array, (arg("array"), arg("dimension")=0), "Slice an array over a dimension and add each slice to an arrayset")

    //some list-like entries
    .def("__len__", &io::Arrayset::size, "The number of arrays stored in this set.")
    .def("__append_array__", (size_t (io::Arrayset::*)(const io::Array&))&io::Arrayset::add, (arg("self"), arg("array")), "Adds an array to this set")
    .def("__append_array__", (size_t (io::Arrayset::*)(boost::shared_ptr<const io::Array>))&io::Arrayset::add, (arg("self"), arg("array")), "Adds an array to this set")

    //some dict-like entries
    .def("__getitem__", (io::Array (io::Arrayset::*)(size_t))&io::Arrayset::operator[], (arg("self"), arg("array_id")), "Gets an array from this set given its id")
    .def("__delitem__", &io::Arrayset::remove, (arg("self"), arg("id")), "Removes the array given its id. May raise an exception if there is no such array inside.")
    .def("__setitem_array__", &pythonic_set<const io::Array>, (arg("self"), arg("id"), arg("array")), "Adds a plain array to this set. If the array-id already exists internally, calling this method will trigger the overwriting of that existing array data.")
    .def("__setitem_array__", &pythonic_set<boost::shared_ptr<const io::Array> >, (arg("self"), arg("id"), arg("array")), "Adds a plain array to this set. If the array-id already exists internally, calling this method will trigger the overwriting of that existing array data.")
    ;

  tp::vector_no_compare<io::Arrayset>("ArraysetVector");
}
