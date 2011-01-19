/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::database::Dataset
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Dataset2.h"

using namespace boost::python;
namespace db = Torch::core;

static const char* get_name(db::Dataset& ds) {
  return ds.getName().c_str();
}

static void set_name(db::Dataset& ds, const char* name) {
  std::string n(name);
  ds.setName(n);
}

void bind_database_dataset() {
  class_<db::Dataset, boost::shared_ptr<db::Dataset> >("Dataset", "Dataset Arrayrepresent lists of Arraysets and Relationsets are bound together to form a homegeneous database description.", init<>("Initializes a new dataset"))
    .def("addArrayset", &db::Dataset::addArrayset, (arg("self"), arg("arrayset")), "Adds an arrayset to this dataset")
    .add_property("name", &get_name, &set_name)
    .add_property("version", &db::Dataset::getVersion, &db::Dataset::setVersion)
    //.def("__len__", &get_narrayset)
    .def("__getitem__", &db::Dataset::getArrayset)
    ;
}
