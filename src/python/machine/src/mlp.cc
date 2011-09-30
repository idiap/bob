/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  7 Jul 15:26:09 2011
 *
 * @brief Python bindings to MLPs
 */

#include <boost/python.hpp>
#include "core/python/vector.h"
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

static blitz::Array<double,1> forward1(const mach::MLP& m,
    const blitz::Array<double,1>& input) {
  blitz::Array<double,1> output(m.outputSize());
  m.forward(input, output);
  return output;
}

static blitz::Array<double,2> forward2(const mach::MLP& m,
    const blitz::Array<double,2>& input) {
  blitz::Array<double,2> output(input.extent(0), m.outputSize());
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

static tuple get_weight(mach::MLP& m) {
  list retval;
  for (std::vector<blitz::Array<double,2> >::const_iterator
      it = m.getWeights().begin(); it != m.getWeights().end(); ++it) {
    retval.append(*it);
  }
  return tuple(retval);
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

static tuple get_bias(mach::MLP& m) {
  list retval;
  for (std::vector<blitz::Array<double,1> >::const_iterator
      it = m.getBiases().begin(); it != m.getBiases().end(); ++it) {
    retval.append(*it);
  }
  return tuple(retval);
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

static void random0(Torch::machine::MLP& M) {
  M.randomize();
}

static void random1(Torch::machine::MLP& M,
    double lower_bound, double upper_bound) {
  M.randomize(lower_bound, upper_bound);
}

static void random2(Torch::machine::MLP& M, boost::mt19937& rng) {
  M.randomize(rng);
}

static void random3(Torch::machine::MLP& M,
   boost::mt19937& rng, double lower_bound, double upper_bound) {
  M.randomize(rng, lower_bound, upper_bound);
}

void bind_machine_mlp() {
  class_<mach::MLP, boost::shared_ptr<mach::MLP>
    >("MLP", "An MLP object is a representation of a Multi-Layer Perceptron. This implementation is feed-forward and fully-connected. The implementation allows setting of input normalization values and a global activation function. References to fully-connected feed-forward networks: Bishop's Pattern Recognition and Machine Learning, Chapter 5. Figure 5.1 shows what we mean.\n\nMLPs normally are multi-layered systems, with 1 or more hidden layers. As a special case, this implementation also supports connecting the input directly to the output by means of a single weight matrix. This is equivalent of a LinearMachine, with the advantage it can be trained by MLP trainers.", init<const std::vector<size_t>&>((arg("shape")), "Builds a new MLP with a shape containing the number of inputs (first element), number of outputs (last element) and the number of neurons in each hidden layer (elements between the first and last element of given tuple). The default activation function will be set to hyperbolic tangent."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new MLP from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency."))
    .def(init<const mach::MLP&>((arg("machine")), "Copy constructs an MLP machine"))
    .def("load", &mach::MLP::load, (arg("self"), arg("config")), "Loads the weights, biases and other configuration parameter sfrom a configuration file.")
    .def("save", &mach::MLP::save, (arg("self"), arg("config")), "Saves the weights and biases to a configuration file.")
    .add_property("input_subtract", make_function(&mach::MLP::getInputSubraction, return_value_policy<copy_const_reference>()), &set_input_sub)
    .add_property("input_divide", make_function(&mach::MLP::getInputDivision, return_value_policy<copy_const_reference>()), &set_input_div)
    .add_property("weights", &get_weight, &set_weight)
    .add_property("biases", &get_bias, &set_bias)
    .add_property("activation", &mach::MLP::getActivation, &mach::MLP::setActivation)
    .add_property("shape", &get_shape, (void (mach::MLP::*)(const std::vector<size_t>&))&mach::MLP::resize)
    .def("__call__", (void (mach::MLP::*)(const blitz::Array<double,1>&, blitz::Array<double,1>&) const)&mach::MLP::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output.")
    .def("forward", (void (mach::MLP::*)(const blitz::Array<double,1>&, blitz::Array<double,1>&) const)&mach::MLP::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output.")
    .def("forward_", (void (mach::MLP::*)(const blitz::Array<double,1>&, blitz::Array<double,1>&) const)&mach::MLP::forward_, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output.")
    .def("__call__", (void (mach::MLP::*)(const blitz::Array<double,2>&, blitz::Array<double,2>&) const)&mach::MLP::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output. This variant will take a number of inputs in one single input matrix with inputs arranged row-wise (i.e., every row contains an individual input).")
    .def("forward", (void (mach::MLP::*)(const blitz::Array<double,2>&, blitz::Array<double,2>&) const)&mach::MLP::forward, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output. This variant will take a number of inputs in one single input matrix with inputs arranged row-wise (i.e., every row contains an individual input).")
    .def("forward_", (void (mach::MLP::*)(const blitz::Array<double,2>&, blitz::Array<double,2>&) const)&mach::MLP::forward_, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output. This variant will take a number of inputs in one single input matrix with inputs arranged row-wise (i.e., every row contains an individual input).")
    .def("__call__", &forward1, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("forward", &forward1, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("__call__", &forward2, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one. This variant will take a number of inputs in one single input matrix with inputs arranged row-wise (i.e., every row contains an individual input).")
    .def("forward", &forward2, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one. This variant will take a number of inputs in one single input matrix with inputs arranged row-wise (i.e., every row contains an individual input).")
    .def("randomize", &random0, (arg("self")), "Sets all weights and biases of this MLP, with random values between [-0.1, 0.1) as advised in textbooks.\n\nValues are drawn using boost::uniform_real class. The seed is picked using a time-based algorithm. Different calls spaced of at least 1 microsecond (machine clock) will be seeded differently. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("randomize", &random1, (arg("self"), arg("lower_bound"), arg("upper_bound")), "Sets all weights and biases of this MLP, with random values between [lower_bound, upper_bound).\n\nValues are drawn using boost::uniform_real class. The seed is picked using a time-based algorithm. Different calls spaced of at least 1 microsecond (machine clock) will be seeded differently. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("randomize", &random2, (arg("self"), arg("rng")), "Sets all weights and biases of this MLP, with random values between [-0.1, 0.1) as advised in textbooks.\n\nValues are drawn using boost::uniform_real class. You should pass the generator in this variant. You can seed it the way it pleases you. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("randomize", &random3, (arg("self"), arg("rng"), arg("lower_bound"), arg("upper_bound")), "Sets all weights and biases of this MLP, with random values between [lower_bound, upper_bound).\n\nValues are drawn using boost::uniform_real class. In this variant you can pass your own random number generate as well as the limits from where the random numbers will be chosen from. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    ;
}
