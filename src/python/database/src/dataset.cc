/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::database::Dataset
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include "database/Dataset.h"

using namespace boost::python;
namespace db = Torch::database;

static const char* get_name(const db::Dataset& ds) {
  return ds.getName().c_str();
}

static void set_name(db::Dataset& ds, const char* name) {
  std::string n(name);
  ds.setName(n);
}

static tuple get_arrayset_ids(db::Dataset& ds) {
  list l;
  for(std::map<size_t, boost::shared_ptr<db::Arrayset> >::const_iterator it=ds.arraysetIndex().begin(); it!=ds.arraysetIndex().end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_arraysets(db::Dataset& ds) {
  list l;
  for(std::map<size_t, boost::shared_ptr<db::Arrayset> >::const_iterator it=ds.arraysetIndex().begin(); it!=ds.arraysetIndex().end(); ++it) {
    l.append(ds.ptr(it->first));
  }
  return tuple(l);
}

static tuple get_relationset_names(db::Dataset& ds) {
  list l;
  for(std::map<std::string, boost::shared_ptr<db::Relationset> >::const_iterator it=ds.relationsetIndex().begin(); it!=ds.relationsetIndex().end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_relationsets(db::Dataset& ds) {
  list l;
  for(std::map<std::string, boost::shared_ptr<db::Relationset> >::const_iterator it=ds.relationsetIndex().begin(); it!=ds.relationsetIndex().end(); ++it) {
    l.append(ds.ptr(it->first));
  }
  return tuple(l);
}

static size_t dataset_set_arrayset(db::Dataset& ds, size_t id, db::Arrayset& as) {
  as.setId(id);
  return ds.add(as);
}

static size_t dataset_set_relationset(db::Dataset& ds, const std::string& name, db::Relationset& rs) {
  rs.setName(name);
  return ds.add(rs);
}

void bind_database_dataset() {
  class_<db::Dataset, boost::shared_ptr<db::Dataset> >("Dataset", "Datasets represent lists of Arraysets and Relationsets that are bound together to form a homegeneous database description.", init<const std::string&, size_t>((arg("name"), arg("version")), "Initializes a new, empty dataset"))
    .def(init<const std::string&>((arg("url")), "Initializes a Dataset reading the contents from a file"))
    .add_property("name", &get_name, &set_name, "The name of this Dataset")
    .add_property("version", &db::Dataset::getVersion, &db::Dataset::setVersion, "The version of this Dataset")
    .def("ids", &get_arrayset_ids, "All Arrayset ids of this Dataset")
    .def("names", &get_relationset_names, "All Relationset names of this Dataset")
    .def("relationsets", &get_relationsets, "All Relationsets of this Dataset")
    .def("arraysets", &get_arraysets, "All Arraysets of this Dataset")
    .def("save", &db::Dataset::save, (arg("self"), arg("path")), "Saves the current dataset into a file representation")
    .def("getNextFreeId", &db::Dataset::getNextFreeId, "Returns the next free arrayset-id")
    .def("consolidateIds", &db::Dataset::consolidateIds, "Re-writes the ids of every arrayset so they are numbered sequentially and by the order of insertion.")

    //appending...
    .def("append", (size_t (db::Dataset::*)(boost::shared_ptr<const db::Arrayset>))&db::Dataset::add, (arg("self"), arg("arrayset")), "Adds an arrayset to this dataset")
    .def("append", (size_t (db::Dataset::*)(boost::shared_ptr<const db::Relationset>))&db::Dataset::add, (arg("self"), arg("relationset")), "Adds a relationset to this dataset")

    //some dictionary-like manipulations
    .def("__getitem__", (boost::shared_ptr<db::Arrayset> (db::Dataset::*)(const size_t))&db::Dataset::ptr, (arg("self"), arg("arrayset_id")), "Returns the Arrayset given its arrayset-id")
    .def("__setitem__", dataset_set_arrayset, (arg("self"), arg("id"), arg("arrayset")), "Sets a given arrayset-id to point to the given arrayset")
    .def("__delitem__", (void (db::Dataset::*)(size_t))&db::Dataset::remove, (arg("self"), arg("arrayset_id")), "Erases a certain arrayset from this dataset")
    .def("__getitem__", (boost::shared_ptr<db::Relationset> (db::Dataset::*)(const std::string&))&db::Dataset::ptr, (arg("self"), arg("relationset_name")), "Returns the Relationset given its relationset-name")
    .def("__setitem__", dataset_set_relationset, (arg("self"), arg("name"), arg("relationset")), "Sets a given relationset-name to point to the given relationset")
    .def("__delitem__", (void (db::Dataset::*)(const std::string&))&db::Dataset::remove, (arg("self"), arg("relationset_name")), "Erases a certain relationset from this dataset")
    ;
}
