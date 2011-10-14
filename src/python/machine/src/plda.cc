/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings for the PLDA{Base,}Machine
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "core/python/exception.h"
#include "core/python/vector.h"
#include "machine/PLDAMachine.h"

using namespace boost::python;
namespace mach = Torch::machine;
namespace io = Torch::io;
namespace tp = Torch::core::python;


void bind_machine_plda() {
  class_<mach::PLDABaseMachine, boost::shared_ptr<mach::PLDABaseMachine> >("PLDABaseMachine", "A PLDABaseMachine", init<const size_t, const size_t, const size_t>((arg("d"), arg("nf"), arg("ng")), "Builds a new PLDABaseMachine. A PLDABaseMachine can be seen as a container for F, G, sigma and mu when performing Probabilistic Linear Discriminant Analysis (PLDA)."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new PLDABaseMachine from a configuration file."))
    .def(init<const mach::PLDABaseMachine&>((arg("machine")), "Copy constructs a PLDABaseMachine"))
    .def("load", &mach::PLDABaseMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &mach::PLDABaseMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("F", make_function(&mach::PLDABaseMachine::getF, return_internal_reference<>()), &mach::PLDABaseMachine::setF)
    .add_property("G", make_function(&mach::PLDABaseMachine::getG, return_internal_reference<>()), &mach::PLDABaseMachine::setG)
    .add_property("sigma", make_function(&mach::PLDABaseMachine::getSigma, return_internal_reference<>()), &mach::PLDABaseMachine::setSigma)
    .add_property("mu", make_function(&mach::PLDABaseMachine::getMu, return_internal_reference<>()), &mach::PLDABaseMachine::setMu)
    .add_property("DimD", &mach::PLDABaseMachine::getDimD)
    .add_property("DimF", &mach::PLDABaseMachine::getDimF)
    .add_property("DimG", &mach::PLDABaseMachine::getDimG)
  ;
}
