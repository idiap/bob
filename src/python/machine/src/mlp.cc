/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  7 Jul 15:26:09 2011
 *
 * @brief Python bindings to MLPs
 */

#include <boost/python.hpp>
#include "core/python/exception.h"
#include "machine/MLP.h"
#include "machine/MLPException.h"

using namespace boost::python;
namespace mach = Torch::machine;
namespace io = Torch::io;
namespace tp = Torch::core::python;

static tuple get_shape(const mach::MLP& m) {
  list retval;
  retval.append(m.inputSize());
  const std::vector<blitz::Array<double,1> >& bias = m.getBiases();
  for (size_t i=0; i<bias.size(); ++i) retval.append(bias[i].extent(0));
  return tuple(retval);
}

static blitz::Array<double,1> forward(const mach::MLP& m,
    const blitz::Array<double,1>& input) {
  blitz::Array<double,1> output(m.outputSize());
  m.forward(input, output);
  return output;
}

static void set_input_sub(mach::MLP& m, object o) {
  extract<int> int_check(o);
  extract<double> float_check(o);
  if (int_check.check()) { //is int
    m.setInputSubtraction(int_check());
  }
  else if (float_check.check()) { //is float
    m.setInputSubtraction(float_check());
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    blitz::Array<double,1> val = extract<blitz::Array<double,1> >(o);
    m.setInputSubtraction(val);
  }
}

static void set_input_div(mach::MLP& m, object o) {
  extract<int> int_check(o);
  extract<double> float_check(o);
  if (int_check.check()) { //is int
    m.setInputDivision(int_check());
  }
  else if (float_check.check()) { //is float
    m.setInputDivision(float_check());
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    blitz::Array<double,1> val = extract<blitz::Array<double,1> >(o);
    m.setInputDivision(val);
  }
}

static void set_weight(mach::MLP& m, object o) {
  extract<int> int_check(o);
  extract<double> float_check(o);
  if (int_check.check()) { //is int
    m.setWeights(int_check());
  }
  else if (float_check.check()) { //is float
    m.setWeights(float_check());
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    m.setWeights(extract<std::vector<blitz::Array<double,2> > >(o));
  }
}

static void set_bias(mach::MLP& m, object o) {
  extract<int> int_check(o);
  extract<double> float_check(o);
  if (int_check.check()) { //is int
    m.setBiases(int_check());
  }
  else if (float_check.check()) { //is float
    m.setBiases(float_check());
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    m.setBiases(extract<std::vector<blitz::Array<double,1> > >(o));
  }
}

void bind_machine_mlp() {
  //exceptions thrown by MLPs
  tp::CxxToPythonTranslator<mach::InvalidShape, mach::Exception>("InvalidShape", "Exception raised when the resizing shape has less than 2 components");
  tp::CxxToPythonTranslatorPar2<mach::NumberOfLayersMismatch, mach::Exception, size_t, size_t>("NumberOfLayersMismatch", "Exception raised when there is a mismatch between the number of layers");
  tp::CxxToPythonTranslatorPar3<mach::WeightShapeMismatch, mach::Exception, size_t, const blitz::TinyVector<int,2>&, const blitz::TinyVector<int,2>&>("WeightShapeMismatch", "Exception raised when there is a mismatch between the shapes of weights to be set and the current MLP size.");
  tp::CxxToPythonTranslatorPar3<mach::BiasShapeMismatch, mach::Exception, size_t, size_t, size_t>("BiasShapeMismatch", "Exception raised when there is a mismatch between the shapes of biases to be set and the current MLP size.");

  class_<mach::MLP, boost::shared_ptr<mach::MLP>
    >("MLP", "An MLP object is a representation of a Multi-Layer Perceptron. This implementation is feed-forward and fully-connected. The implementation allows setting of input normalization values and a global activation function. References to fully-connected feed-forward networks: Bishop's Pattern Recognition and Machine Learning, Chapter 5. Figure 5.1 shows what we mean.\n\nMLPs normally are multi-layered systems, with 1 or more hidden layers. As a special case, this implementation also supports connecting the input directly to the output by means of a single weight matrix. This is equivalent of a LinearMachine, with the advantage it can be trained by MLP trainers.", init<const std::vector<size_t>&>((arg("shape")), "Builds a new MLP with a shape containing the number of inputs (first element), number of outputs (last element) and the number of neurons in each hidden layer (elements between the first and last element of given tuple). The default activation function will be set to hyperbolic tangent."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new MLP from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency."))
    .def("load", &mach::MLP::load, (arg("self"), arg("config")), "Loads the weights, biases and other configuration parameter sfrom a configuration file.")
    .def("save", &mach::MLP::save, (arg("self"), arg("config")), "Saves the weights and biases to a configuration file.")
    .add_property("input_subtract", make_function(&mach::MLP::getInputSubraction, return_internal_reference<>()), &set_input_sub)
    .add_property("input_divide", make_function(&mach::MLP::getInputDivision, return_internal_reference<>()), &set_input_div)
    .add_property("weights", make_function(&mach::MLP::getWeights, return_internal_reference<>()), &set_weight)
    .add_property("biases", make_function(&mach::MLP::getBiases, return_internal_reference<>()), &set_bias)
    .add_property("activation", &mach::MLP::getActivation, &mach::MLP::setActivation)
    .add_property("shape", &get_shape, (void (mach::MLP::*)(const std::vector<size_t>&))&mach::MLP::resize)
    .def("__call__", &mach::MLP::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("forward", &mach::MLP::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("__call__", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("forward", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    ;
}
