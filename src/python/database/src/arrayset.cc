/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::database::Arrayset
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include "core/Dataset2.h"

using namespace boost::python;
namespace db = Torch::core;
namespace dba = Torch::core::array;

tuple get_shape(const db::Arrayset& as) {
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

static void set_shape(db::Arrayset& as, tuple& t) {
  size_t ndim = len(t);
  if (ndim == 0 || ndim > 4) {
    boost::format s("input object must contain from 1 to 4 elements, not %d");
    s % ndim;
    PyErr_SetString(PyExc_RuntimeError, s.str().c_str());
    boost::python::throw_error_already_set();
  }
}

static const char* get_role(db::Arrayset& as) {
  return as.getRole().c_str();
}

static void set_role(db::Arrayset& as, const char* role) {
  std::string r(role);
  as.setRole(r);
}

static const char* get_filename(db::Arrayset& as) {
  return as.getFilename().c_str();
}

static void set_filename(db::Arrayset& as, const char* filename) {
  std::string f(filename);
  as.setFilename(f);
}

static tuple get_array_ids(const db::Arrayset& as) {
  list l;
  for(db::Arrayset::const_iterator it=as.begin(); it!=as.end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_arrays(const db::Arrayset& as) {
  list l;
  for(db::Arrayset::const_iterator it=as.begin(); it!=as.end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

void bind_database_arrayset() {
  enum_<dba::ArrayType>("ArrayType")
    .value("unknown", dba::t_unknown)
    .value("bool", dba::t_bool)
    .value("int8", dba::t_int8)
    .value("int16", dba::t_int16)
    .value("int32", dba::t_int32)
    .value("int64", dba::t_int64)
    .value("uint8", dba::t_uint8)
    .value("uint16", dba::t_uint16)
    .value("uint32", dba::t_uint32)
    .value("uint64", dba::t_uint64)
    .value("float32", dba::t_float32)
    .value("float64", dba::t_float64)
    .value("complex64", dba::t_complex64)
    .value("complex128", dba::t_complex128)
    ;

  class_<db::Arrayset, boost::shared_ptr<db::Arrayset> >("Arrayset", "Dataset Arraysets represent lists of Arrays that share the same element type and dimension properties and are grouped together by the DB designer.", init<>("Initializes a new arrayset"))
    .def("addArray", &db::Arrayset::addArray, (arg("self"), arg("array")), "Adds an array to this set")
    .add_property("id", &db::Arrayset::getId, &db::Arrayset::setId)
    .add_property("shape", &get_shape, &set_shape, "The shape of each array in this set is determined by this variable.")
    .add_property("role", &get_role, &set_role, "This variable determines the role of this arrayset inside this dataset.")
    .add_property("loaded", &db::Arrayset::getIsLoaded, &db::Arrayset::setIsLoaded, "This variable determines if the arrayset is loaded in memory. This may be false in the case the arrayset is completely stored in an external file.")
    .add_property("filename", &get_filename, &set_filename)
    .add_property("arrayType", &db::Arrayset::getArrayType, &db::Arrayset::setArrayType, "This property indicates the type of element used for each array in the current set.")
    .add_property("arrays", &get_arrays, "Iterable over all arrays in this set")
    .add_property("arrayIds", &get_array_ids)
    .def("__len__", &db::Arrayset::getNArrays, "The number of arrays stored in this set.")
    //TODO: Missing __delitem__, __setitem__
    .def("__getitem__", &db::Arrayset::getArray)
    ;
}
