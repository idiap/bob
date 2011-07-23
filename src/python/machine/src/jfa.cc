/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to JFAMachine
 */

#include <boost/python.hpp>
#include "core/python/exception.h"
#include "core/python/vector.h"
#include "machine/JFAMachine.h"

using namespace boost::python;
namespace mach = Torch::machine;
namespace io = Torch::io;
namespace tp = Torch::core::python;


void bind_machine_jfa() {
  class_<mach::JFAMachine, boost::shared_ptr<mach::JFAMachine>
    >("JFAMachine", "A JFAMachine", init<int, int, int, int>((arg("C"), arg("D"), arg("ru"), arg("rv")), "Builds a new JFAMachine."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new JFAMachine from a configuration file."))
    .def(init<const mach::JFAMachine&>((arg("machine")), "Copy constructs a JFAMachine"))
    .def("load", &mach::JFAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .add_property("ubm_mean", make_function(&mach::JFAMachine::getUbmMean, return_internal_reference<>()), &mach::JFAMachine::setUbmMean)
    .add_property("ubm_var", make_function(&mach::JFAMachine::getUbmVar, return_internal_reference<>()), &mach::JFAMachine::setUbmVar)
    .add_property("U", make_function(&mach::JFAMachine::getU, return_internal_reference<>()), &mach::JFAMachine::setU)
    .add_property("V", make_function(&mach::JFAMachine::getV, return_internal_reference<>()), &mach::JFAMachine::setV)
    .add_property("D", make_function(&mach::JFAMachine::getD, return_internal_reference<>()), &mach::JFAMachine::setD)
    .add_property("DimC", &mach::JFAMachine::getDimC)
    .add_property("DimD", &mach::JFAMachine::getDimD)
    .add_property("DimCD", &mach::JFAMachine::getDimCD)
    .add_property("DimRu", &mach::JFAMachine::getDimRu)
    .add_property("DimRv", &mach::JFAMachine::getDimRv)
  ;
}
