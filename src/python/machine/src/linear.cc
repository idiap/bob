/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 31 May 2011 13:29:08 CEST
 *
 * @brief Bindings for a LinearMachine
 */

#include <boost/python.hpp>
#include "machine/LinearMachine.h"

using namespace boost::python;
namespace mach = Torch::machine;
namespace conf = Torch::config;

static blitz::Array<double,1> forward(const mach::LinearMachine& m,
    const blitz::Array<double,1>& input) {
  blitz::Array<double,1> output(m.getBiases().extent(0));
  m.forward(input, output);
  return output;
}

void bind_machine_linear() {
  enum_<mach::LinearMachine::Activation>("Activation")
    .value("LINEAR", mach::LinearMachine::LINEAR)
    .value("TANH", mach::LinearMachine::TANH)
    .value("LOG", mach::LinearMachine::LOG)
    ;

  class_<mach::LinearMachine, boost::shared_ptr<mach::LinearMachine>
    >("LinearMachine", "A linear classifier. See C. M. Bishop, 'Pattern Recognition and Machine  Learning', chapter 4 for more details", init<size_t,size_t>((arg("input_size"), arg("output_size")), "Constructs a new linear machine with a certain input and output sizes. The weights and biases are initialized to zero."))
    .def(init<const blitz::Array<double,2>&, const blitz::Array<double,1>&>((arg("weights"),arg("biases")), "Constructs a new LinearMachine from a set of weights and biases values. Both weights and biases have their dimensionalities checked between each other for consistency."))
    .def(init<const conf::Configuration&>((arg("config")), "Constructs a new LinearMachine from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency."))
    .def("load", &mach::LinearMachine::load, (arg("self"), arg("config")), "Loads the weights and biases from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency.")
    .def("save", &mach::LinearMachine::save, (arg("self"), arg("config")), "Saves the weights and biases to a configuration file.")
    .add_property("input_subtract", &mach::LinearMachine::getInputSubraction, &mach::LinearMachine::setInputSubtraction)
    .add_property("input_divide", &mach::LinearMachine::getInputDivision, &mach::LinearMachine::setInputDivision)
    .add_property("weights", &mach::LinearMachine::getWeights, &mach::LinearMachine::setWeights)
    .add_property("biases", &mach::LinearMachine::getBiases, &mach::LinearMachine::setBiases)
    .add_property("activation", &mach::LinearMachine::getActivation, &mach::LinearMachine::setActivation)
    .def("setAll", &mach::LinearMachine::setAll,
        (arg("self"), arg("input_subtract"), arg("input_divide"), arg("weights"), arg("biases")), "Sets the weights, biases and input normalization parameters of the current machine, checking dimensions so they are compatible with each other.")
    .def("__call__", &mach::LinearMachine::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("forward", &mach::LinearMachine::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("__call__", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("forward", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    ;
}
