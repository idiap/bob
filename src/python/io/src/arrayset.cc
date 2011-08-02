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

template<typename T>
static void add_2d_array_(io::Arrayset& arrayset, const blitz::Array<T, 2>& array_2d) {
  for (int i = 0; i < array_2d.extent(0); i++) {
    blitz::Array<T, 1> tmp = array_2d(i, blitz::Range::all());
    arrayset.add(tmp);
  }
}

static void add_2d_array(io::Arrayset& arrayset, io::Array& array_2d) {

  switch (array_2d.getElementType()) {
    case array::t_bool:
      add_2d_array_(arrayset, array_2d.get<bool, 2>());
      break;
    case array::t_int8:
      add_2d_array_(arrayset, array_2d.get<int8_t, 2>());
      break;
    case array::t_int16:
      add_2d_array_(arrayset, array_2d.get<int16_t, 2>());
      break;
    case array::t_int32:
      add_2d_array_(arrayset, array_2d.get<int32_t, 2>());
      break;
    case array::t_uint8:
      add_2d_array_(arrayset, array_2d.get<uint8_t, 2>());
      break;
    case array::t_uint16:
      add_2d_array_(arrayset, array_2d.get<uint16_t, 2>());
      break;
    case array::t_uint32:
      add_2d_array_(arrayset, array_2d.get<uint32_t, 2>());
      break;
    case array::t_uint64:
      add_2d_array_(arrayset, array_2d.get<uint64_t, 2>());
      break;
    case array::t_float32:
      add_2d_array_(arrayset, array_2d.get<float, 2>());
      break;
    case array::t_float64:
      add_2d_array_(arrayset, array_2d.get<double, 2>());
      break;
    case array::t_float128:
      add_2d_array_(arrayset, array_2d.get<long double, 2>());
      break;
    case array::t_complex64:
      add_2d_array_(arrayset, array_2d.get<std::complex<float>, 2>());
      break;
    case array::t_complex128:
      add_2d_array_(arrayset, array_2d.get<std::complex<double>, 2>());
      break;
    case array::t_complex256:
      add_2d_array_(arrayset, array_2d.get<std::complex<long double>, 2>());
      break;

    default:
      throw io::TypeError(array_2d.getElementType(), array::getElementType<double>());
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
    .def("append_2d_array", &add_2d_array, "Add each line of a 2d array to the Arrayset")

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
