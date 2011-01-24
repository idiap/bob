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

using namespace boost::python;
namespace db = Torch::core;

static const char* get_name(const db::Dataset& ds) {
  return ds.getName().c_str();
}

static void set_name(db::Dataset& ds, const char* name) {
  std::string n(name);
  ds.setName(n);
}

static boost::shared_ptr<db::Dataset> loadxml(const char* url) {
  Torch::core::XMLParser parser;
  boost::shared_ptr<db::Dataset> retval(new db::Dataset);
  parser.load(url, *retval);
  return retval;
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
  class_<db::Dataset, boost::shared_ptr<db::Dataset> >("Dataset", "Datasets represent lists of Arraysets and Relationsets that are bound together to form a homegeneous database description.", init<>("Initializes a new dataset"))
    .def("addArrayset", &db::Dataset::addArrayset, (arg("self"), arg("arrayset")), "Adds an arrayset to this dataset")
    .add_property("name", &get_name, &set_name, "The name of this Dataset")
    .add_property("version", &db::Dataset::getVersion, &db::Dataset::setVersion, "The version of this Dataset")
    .add_property("arraysets", &get_arraysets, "All Arraysets of this Dataset")
    .add_property("relationsets", &get_relationsets, "All Relationsets of this Dataset")
    ;

  def("__load_local_xml__", &loadxml, (arg("path")), "Loads a local Dataset stored in an XML file");
}
