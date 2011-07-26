/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to JFAMachine
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "core/python/exception.h"
#include "core/python/vector.h"
#include "machine/JFAMachine.h"
#include "machine/GMMMachine.h"

using namespace boost::python;
namespace mach = Torch::machine;
namespace io = Torch::io;
namespace tp = Torch::core::python;


void bind_machine_jfa() {
  class_<mach::JFABaseMachine, boost::shared_ptr<mach::JFABaseMachine> >("JFABaseMachine", "A JFABaseMachine", init<boost::shared_ptr<mach::GMMMachine>, int, int>((arg("ubm"), arg("ru"), arg("rv")), "Builds a new JFABaseMachine."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new JFABaseMachine from a configuration file."))
    .def(init<const mach::JFABaseMachine&>((arg("machine")), "Copy constructs a JFABaseMachine"))
    .def("load", &mach::JFABaseMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .add_property("ubm", &mach::JFABaseMachine::getUbm, &mach::JFABaseMachine::setUbm)
    .add_property("U", make_function(&mach::JFABaseMachine::getU, return_internal_reference<>()), &mach::JFABaseMachine::setU)
    .add_property("V", make_function(&mach::JFABaseMachine::getV, return_internal_reference<>()), &mach::JFABaseMachine::setV)
    .add_property("D", make_function(&mach::JFABaseMachine::getD, return_internal_reference<>()), &mach::JFABaseMachine::setD)
    .add_property("DimC", &mach::JFABaseMachine::getDimC)
    .add_property("DimD", &mach::JFABaseMachine::getDimD)
    .add_property("DimCD", &mach::JFABaseMachine::getDimCD)
    .add_property("DimRu", &mach::JFABaseMachine::getDimRu)
    .add_property("DimRv", &mach::JFABaseMachine::getDimRv)
  ;

  class_<mach::JFAMachine, boost::shared_ptr<mach::JFAMachine> >("JFAMachine", "A JFAMachine", init<boost::shared_ptr<mach::JFABaseMachine> >((arg("jfa_base")), "Builds a new JFAMachine."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new JFAMachine from a configuration file."))
    .def(init<const mach::JFAMachine&>((arg("machine")), "Copy constructs a JFAMachine"))
    .def("load", &mach::JFAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .add_property("ubm", &mach::JFAMachine::getJFABase, &mach::JFAMachine::setJFABase)
    .add_property("y", make_function(&mach::JFAMachine::getY, return_internal_reference<>()), &mach::JFAMachine::setY)
    .add_property("z", make_function(&mach::JFAMachine::getZ, return_internal_reference<>()), &mach::JFAMachine::setZ)
    .add_property("DimC", &mach::JFAMachine::getDimC)
    .add_property("DimD", &mach::JFAMachine::getDimD)
    .add_property("DimCD", &mach::JFAMachine::getDimCD)
    .add_property("DimRu", &mach::JFAMachine::getDimRu)
    .add_property("DimRv", &mach::JFAMachine::getDimRv)
  ;

}
