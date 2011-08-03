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

// Partial loop specializations for extending Arraysets 
// with higher dimensional Arrays
template <typename T, int N> struct looper {
  static void call (io::Arrayset& self, const io::Array& A, int D) {
    PYTHON_ERROR(RuntimeError, "unsupported generic looper");
  }
};

template <typename T> struct looper<T,2> {
  static void call (io::Arrayset& self, const io::Array& A, int D) {
    if (D > 1 || D < 0) PYTHON_ERROR(RuntimeError, "bad dimension index");
    const blitz::Array<T,2> bz = A.get<T,2>();
    blitz::Range all = blitz::Range::all();
    switch (D) {
      case 0:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(i,all));
        break;
      case 1:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(all,i));
        break;
    }
  }
};

template <typename T> struct looper<T,3> {
  static void call (io::Arrayset& self, const io::Array& A, int D) {
    if (D > 2 || D < 0) PYTHON_ERROR(RuntimeError, "bad dimension index");
    const blitz::Array<T,3> bz = A.get<T,3>();
    blitz::Range all = blitz::Range::all();
    switch (D) {
      case 0:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(i,all,all));
        break;
      case 1:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(all,i,all));
        break;
      case 2:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(all,all,i));
        break;
    }
  }
};

template <typename T> struct looper<T,4> {
  static void call (io::Arrayset& self, const io::Array& A, int D) {
    if (D > 3 || D < 0) PYTHON_ERROR(RuntimeError, "bad dimension index");
    const blitz::Array<T,4> bz = A.get<T,4>();
    blitz::Range all = blitz::Range::all();
    switch (D) {
      case 0:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(i,all,all,all));
        break;
      case 1:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(all,i,all,all));
        break;
      case 2:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(all,all,i,all));
        break;
      case 3:
        for (int i=0; i<bz.extent(D); ++i) self.add(bz(all,all,all,i));
        break;
    }
  }
};

// Switch case for the number of dimensions available at the input array
#define DIM_SWITCH(T) \
switch (array.getNDim()) {\
  case 2: \
    looper<T,2>::call(self, array, D); \
    break; \
  case 3: \
    looper<T,3>::call(self, array, D); \
    break; \
  case 4: \
    looper<T,4>::call(self, array, D); \
    break; \
default: \
    PYTHON_ERROR(RuntimeError, "unsupported number of dimensions for extend"); \
}

static void array_extend(io::Arrayset& self, const io::Array& array, int D) {
  switch (array.getElementType()) {
    case array::t_bool:       DIM_SWITCH(bool);                 break;
    case array::t_int8:       DIM_SWITCH(int8_t);               break;
    case array::t_int16:      DIM_SWITCH(int16_t);              break;
    case array::t_int32:      DIM_SWITCH(int32_t);              break;
    case array::t_int64:      DIM_SWITCH(int64_t);              break;
    case array::t_uint8:      DIM_SWITCH(uint8_t);              break;
    case array::t_uint16:     DIM_SWITCH(uint16_t);             break;
    case array::t_uint32:     DIM_SWITCH(uint32_t);             break;
    case array::t_uint64:     DIM_SWITCH(uint64_t);             break;
    case array::t_float32:    DIM_SWITCH(float);                break;
    case array::t_float64:    DIM_SWITCH(double);               break;
    case array::t_complex64:  DIM_SWITCH(std::complex<float>);  break;
    case array::t_complex128: DIM_SWITCH(std::complex<double>); break;
    default:
      PYTHON_ERROR(RuntimeError, "unsupported element type for extend");
  }
}

template <typename T, int N>
static void vector_extend(io::Arrayset& self,
    const std::vector<blitz::Array<T,N> >& v) {
  for (size_t i=0; i<v.size(); ++i) self.add(v[i]);
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
    .def("save", &io::Arrayset::save, arrayset_save_overloads((arg("self"), arg("filename"), arg("codecname")=""), "Saves, renames or re-writes the arrayset into a file. It will save if the arrayset is loaded in memory. It will move if the codec used does not change by the filename does. It will re-write if the codec changes."))
    .def("load", &io::Arrayset::load)
    .def("__array_extend__", &array_extend, (arg("self"), arg("array"), arg("dimension")), "Slice an array over a dimension and add each slice to the arrayset")
#   define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
#   define BOOST_PP_LOCAL_MACRO(D) \
    .def("__iterable_extend__", &vector_extend<bool,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<int8_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<int16_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<int32_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<int64_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<uint8_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<uint16_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<uint32_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<uint64_t,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<float,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<double,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<std::complex<float>,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset") \
    .def("__iterable_extend__", &vector_extend<std::complex<double>,D>, (arg("self"), arg("iterable")), "Adds a bunch of arrays to this arrayset")
#   include BOOST_PP_LOCAL_ITERATE()

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
