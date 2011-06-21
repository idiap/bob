/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::database::Arrayset
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>

#include "database/Arrayset.h"
#include "core/python/vector.h"

using namespace boost::python;
namespace db = Torch::database;
namespace core = Torch::core;
namespace array = Torch::core::array;
namespace tp = Torch::python;

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

static tuple get_array_ids(const db::Arrayset& as) {
  std::vector<size_t> t;
  as.index(t);
  list l;
  for(std::vector<size_t>::iterator it=t.begin(); it!=t.end(); ++it) {
    l.append(*it);
  }
  return tuple(l);
}

template <typename T>
static void pythonic_set (db::Arrayset& as, size_t id, T obj) {
  if (as.exists(id)) as.set(id, obj);
  else as.add(id, obj);
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(arrayset_save_overloads, save, 1, 2) 

void bind_database_arrayset() {
  class_<db::Arrayset, boost::shared_ptr<db::Arrayset> >("Arrayset", "Dataset Arraysets represent lists of Arrays that share the same element type and dimension properties and are grouped together by the DB designer.", init<const std::string&, optional<const std::string&> >((arg("filename"),arg("codecname")=""), "Initializes a new arrayset from an external file. An optional codec may be passed."))
    .def(init<>("Creates a new empty arraset with an inlined representation."))
    .add_property("shape", &get_shape, "The shape of each array in this set is determined by this variable.")
    .add_property("role", &get_role, &set_role, "This variable determines the role of this arrayset inside this dataset.")
    .add_property("loaded", &db::Arrayset::isLoaded, "This variable determines if the arrayset is loaded in memory. This may be false in the case the arrayset is completely stored in an external file.")
    .add_property("filename", &get_filename)
    .add_property("elementType", &db::Arrayset::getElementType, "This property indicates the type of element used for each array in the current set.")
    .def("save", &db::Arrayset::save, arrayset_save_overloads((arg("filename"), arg("codecname")=""), "Saves, renames or re-writes the arrayset into a file. It will save if the arrayset is loaded in memory. It will move if the codec used does not change by the filename does. It will re-write if the codec changes."))
    .def("load", &db::Arrayset::load)
    .def("consolidateIds", &db::Arrayset::consolidateIds, "Re-numbers all ids so they are sequential and starting at 1")

    //some list-like entries
    .def("__len__", &db::Arrayset::getNSamples, "The number of arrays stored in this set.")
    .def("__append_array__", (size_t (db::Arrayset::*)(const db::Array&))&db::Arrayset::add, (arg("self"), arg("array")), "Adds an array to this set")
    .def("__append_array__", (size_t (db::Arrayset::*)(boost::shared_ptr<const db::Array>))&db::Arrayset::add, (arg("self"), arg("array")), "Adds an array to this set")

    //some dict-like entries
    .def("ids", &get_array_ids, "The ids of every array in this set, in a tuple")
    .def("exists", &db::Arrayset::exists, (arg("self"), arg("array_id")), "Returns True if I have an Array with the given array-id") 
    .def("__getitem__", (db::Array (db::Arrayset::*)(size_t))&db::Arrayset::operator[], (arg("self"), arg("array_id")), "Gets an array from this set given its id")
    .def("__delitem__", &db::Arrayset::remove, (arg("self"), arg("id")), "Removes the array given its id. May raise an exception if there is no such array inside.")
    .def("__setitem_array__", &pythonic_set<const db::Array>, (arg("self"), arg("id"), arg("array")), "Adds a plain array to this set. If the array-id already exists internally, calling this method will trigger the overwriting of that existing array data.")
    .def("__setitem_array__", &pythonic_set<boost::shared_ptr<const db::Array> >, (arg("self"), arg("id"), arg("array")), "Adds a plain array to this set. If the array-id already exists internally, calling this method will trigger the overwriting of that existing array data.")
    ;

  tp::vector_no_compare<db::Arrayset>("ArraysetVector");
}
