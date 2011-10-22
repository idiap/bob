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
    .add_property("alpha", make_function(&mach::PLDABaseMachine::getAlpha, return_internal_reference<>()))
    .add_property("beta", make_function(&mach::PLDABaseMachine::getBeta, return_internal_reference<>()))
    .def("computeGamma", &mach::PLDABaseMachine::computeGamma, (arg("self"), arg("a"), arg("gamma")), "Computes a gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .add_property("DimD", &mach::PLDABaseMachine::getDimD)
    .add_property("DimF", &mach::PLDABaseMachine::getDimF)
    .add_property("DimG", &mach::PLDABaseMachine::getDimG)
  ;

  class_<mach::PLDAMachine, boost::shared_ptr<mach::PLDAMachine> >("PLDAMachine", "A PLDAMachine", init<boost::shared_ptr<mach::PLDABaseMachine> >((arg("plda_base")), "Builds a new PLDAMachine. An attached PLDABaseMachine should be provided, containing the PLDA model (F, G and Sigma). The PLDAMachine only carries information the enrolled samples."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new PLDAMachine from a configuration file."))
    .def(init<const mach::PLDAMachine&>((arg("machine")), "Copy constructs a PLDAMachine"))
    .def("load", &mach::PLDAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &mach::PLDAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("computeLikelihood", &mach::PLDAMachine::computeLikelihood, (arg("self"), arg("arrayset")), "Computes the likelihood considering jointly the samples contained in the given Arrayset and the enrolled samples.")
//    .def("__call__", &jfa_forward_sample, (arg("self"), arg("gmm_stats")), "Processes GMM statistics and returns a score.")
//    .def("forward", &jfa_forward_sample, (arg("self"), arg("gmm_stats")), "Processes GMM statistics and returns a score.")
//    .def("__call__", &jfa_forward_list, (arg("self"), arg("gmm_stats"), arg("scores")), "Processes a list of GMM statistics and updates a score list.")
//    .def("forward", &jfa_forward_list, (arg("self"), arg("gmm_stats"), arg("scores")), "Processes a list of GMM statistics and updates a score list.")
    .add_property("plda_base", &mach::PLDAMachine::getPLDABase, &mach::PLDAMachine::setPLDABase)
//    .add_property("enrolledSamples", make_function(&mach::PLDAMachine::getEnrolledSamples, return_internal_reference<>()), &mach::PLDAMachine::setEnrolledSamples)
    .add_property("DimD", &mach::PLDAMachine::getDimD)
    .add_property("DimF", &mach::PLDAMachine::getDimF)
    .add_property("DimG", &mach::PLDAMachine::getDimG)
  ;



}
