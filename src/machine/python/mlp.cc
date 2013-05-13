/**
 * @file machine/python/mlp.cc
 * @date Thu Jul 7 16:49:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to MLPs
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <bob/core/python/ndarray.h>
#include <boost/make_shared.hpp>
#include <boost/python/stl_iterator.hpp>
#include <bob/machine/MLP.h>
#include <bob/machine/MLPException.h>

using namespace boost::python;

static tuple get_shape(const bob::machine::MLP& m) {
  list retval;
  retval.append(m.inputSize());
  const std::vector<blitz::Array<double,1> >& bias = m.getBiases();
  for (size_t i=0; i<bias.size(); ++i) retval.append(bias[i].extent(0));
  return tuple(retval);
}

static void set_shape(bob::machine::MLP& m, object shape) {
  stl_input_iterator<size_t> begin(shape), end;
  std::vector<size_t> vshape(begin, end);
  m.resize(vshape);
}

static object forward1(bob::machine::MLP& m, bob::python::const_ndarray input) {

  const bob::core::array::typeinfo& info = input.type();

  if (info.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, "cannot forward arrays of type '%s'", info.str().c_str());

  switch(info.nd) {
    case 1:
      {
        bob::python::ndarray output(bob::core::array::t_float64, m.outputSize());
        blitz::Array<double,1> output_ = output.bz<double,1>();
        m.forward(input.bz<double,1>(), output_);
        return output.self();
      }
      break;
    case 2:
      {
        bob::python::ndarray output(bob::core::array::t_float64, input.type().shape[0],m.outputSize());
        blitz::Array<double,2> output_ = output.bz<double,2>();
        m.forward(input.bz<double,2>(), output_);
        return output.self();
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "cannot forward arrays of type '%s'", info.str().c_str());
  }
}

static void forward2(bob::machine::MLP& m, bob::python::const_ndarray input,
    bob::python::ndarray output) {
  const bob::core::array::typeinfo& info = input.type();

  if (info.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, "cannot forward arrays of type '%s'", info.str().c_str());

  switch(info.nd) {
    case 1:
      {
        blitz::Array<double,1> output_ = output.bz<double,1>();
        m.forward(input.bz<double,1>(), output_);
      }
      break;
    case 2:
      {
        blitz::Array<double,2> output_ = output.bz<double,2>();
        m.forward(input.bz<double,2>(), output_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "cannot forward arrays of type '%s'", info.str().c_str());
  }
}

static void forward2_(bob::machine::MLP& m, bob::python::const_ndarray input,
    bob::python::ndarray output) {
  const bob::core::array::typeinfo& info = input.type();

  if (info.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, "cannot forward arrays of type '%s'", info.str().c_str());

  switch(info.nd) {
    case 1:
      {
        blitz::Array<double,1> output_ = output.bz<double,1>();
        m.forward_(input.bz<double,1>(), output_);
      }
      break;
    case 2:
      {
        blitz::Array<double,2> output_ = output.bz<double,2>();
        m.forward_(input.bz<double,2>(), output_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "cannot forward arrays of type '%s'", info.str().c_str());
  }
}

static void set_input_sub(bob::machine::MLP& m, object o) {
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

static void set_input_div(bob::machine::MLP& m, object o) {
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

static tuple get_weight(bob::machine::MLP& m) {
  list retval;
  for (std::vector<blitz::Array<double,2> >::const_iterator
      it = m.getWeights().begin(); it != m.getWeights().end(); ++it) {
    retval.append(*it);
  }
  return tuple(retval);
}

static void set_weight(bob::machine::MLP& m, object o) {
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
    stl_input_iterator<blitz::Array<double,2> > begin(o), end;
    std::vector<blitz::Array<double,2> > vec(begin, end);
    m.setWeights(vec);
  }
}

static tuple get_bias(const bob::machine::MLP& m) {
  list retval;
  for (std::vector<blitz::Array<double,1> >::const_iterator
      it = m.getBiases().begin(); it != m.getBiases().end(); ++it) {
    retval.append(*it);
  }
  return tuple(retval);
}

static void set_bias(bob::machine::MLP& m, object o) {
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
    stl_input_iterator<blitz::Array<double,1> > begin(o), end;
    std::vector<blitz::Array<double,1> > vec(begin, end);
    m.setBiases(vec);
  }
}

static tuple get_z(const bob::machine::MLP& m) {
  list retval;
  for (std::vector<blitz::Array<double,1> >::const_iterator
      it = m.getZ().begin(); it != m.getZ().end(); ++it) {
    retval.append(*it);
  }
  return tuple(retval);
}

static tuple get_a(const bob::machine::MLP& m) {
  list retval;
  for (std::vector<blitz::Array<double,1> >::const_iterator
      it = m.getA().begin(); it != m.getA().end(); ++it) {
    retval.append(*it);
  }
  return tuple(retval);
}

static tuple get_b(const bob::machine::MLP& m) {
  list retval;
  for (std::vector<blitz::Array<double,1> >::const_iterator
      it = m.getB().begin(); it != m.getB().end(); ++it) {
    retval.append(*it);
  }
  return tuple(retval);
}

static void random0(bob::machine::MLP& M) {
  M.randomize();
}

static void random1(bob::machine::MLP& M,
    double lower_bound, double upper_bound) {
  M.randomize(lower_bound, upper_bound);
}

static void random2(bob::machine::MLP& M, boost::mt19937& rng) {
  M.randomize(rng);
}

static void random3(bob::machine::MLP& M,
   boost::mt19937& rng, double lower_bound, double upper_bound) {
  M.randomize(rng, lower_bound, upper_bound);
}

static boost::shared_ptr<bob::machine::MLP> mlp_from_shape(object shape) {
  stl_input_iterator<size_t> begin(shape), end;
  std::vector<size_t> vshape(begin, end);
  return boost::make_shared<bob::machine::MLP>(vshape);
}

void bind_machine_mlp() {
  class_<bob::machine::MLP, boost::shared_ptr<bob::machine::MLP> >("MLP", "An MLP object is a representation of a Multi-Layer Perceptron. This implementation is feed-forward and fully-connected. The implementation allows setting of input normalization values and a global activation function. References to fully-connected feed-forward networks: Bishop's Pattern Recognition and Machine Learning, Chapter 5. Figure 5.1 shows what we mean.\n\nMLPs normally are multi-layered systems, with 1 or more hidden layers. As a special case, this implementation also supports connecting the input directly to the output by means of a single weight matrix. This is equivalent of a LinearMachine, with the advantage it can be trained by MLP trainers.", no_init)
    .def("__init__", make_constructor(&mlp_from_shape, default_call_policies(), (arg("shape"))), "Builds a new MLP with a shape containing the number of inputs (first element), number of outputs (last element) and the number of neurons in each hidden layer (elements between the first and last element of given tuple). The default activation function will be set to hyperbolic tangent.")
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new MLP from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency."))
    .def(init<const bob::machine::MLP&>((arg("machine")), "Copy constructs an MLP machine"))
    .def("load", &bob::machine::MLP::load, (arg("self"), arg("config")), "Loads the weights, biases and other configuration parameter sfrom a configuration file.")
    .def("save", &bob::machine::MLP::save, (arg("self"), arg("config")), "Saves the weights and biases to a configuration file.")
    .add_property("input_subtract", make_function(&bob::machine::MLP::getInputSubtraction, return_value_policy<copy_const_reference>()), &set_input_sub, "Input subtraction factor, before feeding data through the MLP. The subtraction is the first applied operation in the processing chain - by default, it is set to 0.0.")
    .add_property("input_divide", make_function(&bob::machine::MLP::getInputDivision, return_value_policy<copy_const_reference>()), &set_input_div, "Input division factor, before feeding data through the MLP. The division is applied just after subtraction - by default, it is set to 1.0")
    .add_property("weights", &get_weight, &set_weight, "A set of weights for the synapses connecting each layer in the MLP. This is represented by a standard tuple containing the weights as 2D numpy.ndarray's of double-precision floating-point elements. Each of the ndarrays has the number of rows equals to the input received by that layer and the number of columns equals to the output fed to the next layer.")
    .add_property("biases", &get_bias, &set_bias, "A set of biases for each layer in the MLP. This is represented by a standard tuple containing the biases as 1D numpy.ndarray's of double-precision floating-point elements. Each of the ndarrays has the number of elements equals to the number of neurons in the respective layer. Note that, by definition, the input layer is not subject to biasing. If you need biasing on the input layer, use the input_subtract and input_divide attributes of this MLP.")
    .add_property("z", &get_z, "The outputs of each (hidden ones and output one) layer before applying the activation function, after using the forward propagation. This is represented by a standard tuple containing the biases as 1D numpy.ndarray's of double-precision floating-point elements. Each of the ndarrays has the number of elements equals to the number of neurons in the respective layer.")
    .add_property("a", &get_a, "The outputs of each (input one and hidden ones) layer after applying the activation function, after using the forward propagation. This is represented by a standard tuple containing the biases as 1D numpy.ndarray's of double-precision floating-point elements. Each of the ndarrays has the number of elements equals to the number of neurons in the respective layer.")
    .add_property("b", &get_b, "The outputs of each (hidden ones and output one) layer before applying the activation function, after using the backward propagation. This is represented by a standard tuple containing the biases as 1D numpy.ndarray's of double-precision floating-point elements. Each of the ndarrays has the number of elements equals to the number of neurons in the respective layer.")
    .add_property("activation", &bob::machine::MLP::getActivation, &bob::machine::MLP::setActivation, "The activation function (for all hidden layers) - by default, the hyperbolic tangent function. The output provided by the activation function is passed, unchanged, to the user.")
    .add_property("output_activation", &bob::machine::MLP::getOutputActivation, &bob::machine::MLP::setOutputActivation, "The output activation function (only for the last output layer) - by default, the hyperbolic tangent function. The output provided by the activation function is passed, unchanged, to the user.")
    .add_property("shape", &get_shape, &set_shape, "A tuple that represents the size of the input vector followed by the number of neurons in each hidden layer of the MLP and, finally, terminated by the size of the output vector in the format ``(input, hidden0, hidden1, ..., hiddenN, output)``. If you set this attribute, the network is automatically resized and should be considered uninitialized.")
    .def("__call__", &forward2, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output. You can either pass an input with 1 or 2 dimensions. If 2D, it is the same as running the 1D case many times considering as input to be every row in the input matrix.")
    .def("forward", &forward2, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output. You can either pass an input with 1 or 2 dimensions. If 2D, it is the same as running the 1D case many times considering as input to be every row in the input matrix.")
    .def("forward_", &forward2_, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output. You can either pass an input with 1 or 2 dimensions. If 2D, it is the same as running the 1D case many times considering as input to be every row in the input matrix.")
    .def("__call__", &forward1, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one. You can either pass an input with 1 or 2 dimensions. If 2D, it is the same as running the 1D case many times considering as input to be every row in the input matrix.")
    .def("forward", &forward1, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one. You can either pass an input with 1 or 2 dimensions. If 2D, it is the same as running the 1D case many times considering as input to be every row in the input matrix.")
    .def("randomize", &random0, (arg("self")), "Sets all weights and biases of this MLP, with random values between [-0.1, 0.1) as advised in textbooks.\n\nValues are drawn using boost::uniform_real class. The seed is picked using a time-based algorithm. Different calls spaced of at least 1 microsecond (machine clock) will be seeded differently. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("randomize", &random1, (arg("self"), arg("lower_bound"), arg("upper_bound")), "Sets all weights and biases of this MLP, with random values between [lower_bound, upper_bound).\n\nValues are drawn using boost::uniform_real class. The seed is picked using a time-based algorithm. Different calls spaced of at least 1 microsecond (machine clock) will be seeded differently. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("randomize", &random2, (arg("self"), arg("rng")), "Sets all weights and biases of this MLP, with random values between [-0.1, 0.1) as advised in textbooks.\n\nValues are drawn using boost::uniform_real class. You should pass the generator in this variant. You can seed it the way it pleases you. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("randomize", &random3, (arg("self"), arg("rng"), arg("lower_bound"), arg("upper_bound")), "Sets all weights and biases of this MLP, with random values between [lower_bound, upper_bound).\n\nValues are drawn using boost::uniform_real class. In this variant you can pass your own random number generate as well as the limits from where the random numbers will be chosen from. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    ;
}
