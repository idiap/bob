/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::database::Dataset
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include "core/Dataset2.h"
#include "core/XMLParser.h"
#include "core/XMLWriter.h"

using namespace boost::python;
namespace db = Torch::core;

static const char* get_name(const db::Dataset& ds) {
  return ds.getName().c_str();
}

static void set_name(db::Dataset& ds, const char* name) {
  std::string n(name);
  ds.setName(n);
}

static boost::shared_ptr<db::Dataset> dataset_from_xml(const char* path) {
  db::XMLParser parser;
  boost::shared_ptr<db::Dataset> retval(new db::Dataset);
  parser.load(path, *retval);
  return retval;
}

static void dataset_to_xml(const db::Dataset& ds, const char* path) {
  db::XMLWriter writer;
  writer.write(path, ds);
}

static tuple get_arrayset_ids(const db::Dataset& ds) {
  list l;
  for(db::Dataset::const_iterator it=ds.begin(); it!=ds.end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_arraysets(const db::Dataset& ds) {
  list l;
  for(db::Dataset::const_iterator it=ds.begin(); it!=ds.end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

static tuple get_relationset_names(const db::Dataset& ds) {
  list l;
  for(db::Dataset::relationset_const_iterator it=ds.relationset_begin(); it!=ds.relationset_end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_relationsets(const db::Dataset& ds) {
  list l;
  for(db::Dataset::relationset_const_iterator it=ds.relationset_begin(); it!=ds.relationset_end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

void bind_database_dataset() {
  class_<db::Dataset, boost::shared_ptr<db::Dataset> >("Dataset", "Datasets represent lists of Arraysets and Relationsets that are bound together to form a homegeneous database description.", init<>("Initializes a new, empty dataset"))
    .def("__init__", make_constructor(dataset_from_xml))
    .def("append", &db::Dataset::addArrayset, (arg("self"), arg("arrayset")), "Adds an arrayset to this dataset")
    .def("append", &db::Dataset::addRelationset, (arg("self"), arg("relationset")), "Adds a relationset to this dataset")
    .def("__getitem__", (boost::shared_ptr<db::Arrayset> (db::Dataset::*)(const size_t))&db::Dataset::getArrayset, (arg("self"), arg("arrayset_id")), "Returns the Arrayset given its arrayset-id")
    .def("__getitem__", (boost::shared_ptr<db::Relationset> (db::Dataset::*)(const std::string&))&db::Dataset::getRelationset, (arg("self"), arg("relationset_name")), "Returns the Relationset given its relationset-name")
    .add_property("name", &get_name, &set_name, "The name of this Dataset")
    .add_property("version", &db::Dataset::getVersion, &db::Dataset::setVersion, "The version of this Dataset")
    .add_property("arraysets", &get_arraysets, "All Arraysets of this Dataset")
    .add_property("relationsets", &get_relationsets, "All Relationsets of this Dataset")
    .def("save", &dataset_to_xml, (arg("self"), arg("path")), "Saves the current dataset into a local file in XML representation")
    ;
}
