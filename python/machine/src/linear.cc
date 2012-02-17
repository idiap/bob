/**
 * @file python/machine/src/linear.cc
 * @date Tue May 31 13:33:31 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Bindings for a LinearMachine
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "core/python/ndarray.h"
#include "machine/LinearMachine.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;
namespace mach = bob::machine;
namespace io = bob::io;

static object forward(const mach::LinearMachine& m, tp::const_ndarray input) {
  tp::ndarray output(ca::t_float64, m.outputSize());
  blitz::Array<double,1> output_ = output.bz<double,1>();
  m.forward(input.bz<double,1>(), output_);
  return output.self();
}

static void forward2(const mach::LinearMachine& m, tp::const_ndarray input,
    tp::ndarray output) {
  blitz::Array<double,1> output_ = output.bz<double,1>();
  m.forward(input.bz<double,1>(), output_);
}

static tuple get_shape(const mach::LinearMachine& m) {
  return make_tuple(m.inputSize(), m.outputSize());
}

static void set_shape(mach::LinearMachine& m,
    const blitz::TinyVector<int,2>& s) {
  m.resize(s(0), s(1));
}

static void set_input_sub(mach::LinearMachine& m, object o) {
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

static void set_input_div(mach::LinearMachine& m, object o) {
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

static void set_weight(mach::LinearMachine& m, object o) {
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
    blitz::Array<double,2> val = extract<blitz::Array<double,2> >(o);
    m.setWeights(val);
  }
}

static void set_bias(mach::LinearMachine& m, object o) {
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
    blitz::Array<double,1> val = extract<blitz::Array<double,1> >(o);
    m.setBiases(val);
  }
}

void bind_machine_linear() {
  class_<mach::LinearMachine, boost::shared_ptr<mach::LinearMachine>
    >("LinearMachine", "A linear classifier. See C. M. Bishop, 'Pattern Recognition and Machine  Learning', chapter 4 for more details.\n\nThe basic matrix operation performed for projecting the input to the output is: output = weights * input. The 'weights' matrix is therefore organized column-wise. In this scheme, each column of the weights matrix can be interpreted as vector to which the input is projected.\n\nThe number of columns of the weights matrix determines the number of outputs this linear machine will have. The number of rows, the number of allowed inputs it can process.", init<size_t,size_t>((arg("input_size"), arg("output_size")), "Constructs a new linear machine with a certain input and output sizes. The weights and biases are initialized to zero."))
    .def(init<const blitz::Array<double,2>&>((arg("weights")), "Constructs a new LinearMachine from a set of weight values. Each column of the weight matrix should represent a direction to which the input is projected."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new LinearMachine from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency."))
    .def(init<>("Default constructor, builds a machine as with 'LinearMachine(0,0)' which, of course, does not accept inputs or produce outputs."))
    .def("load", &mach::LinearMachine::load, (arg("self"), arg("config")), "Loads the weights and biases from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency.")
    .def("save", &mach::LinearMachine::save, (arg("self"), arg("config")), "Saves the weights and biases to a configuration file.")
    .add_property("input_subtract", make_function(&mach::LinearMachine::getInputSubraction, return_value_policy<copy_const_reference>()), &set_input_sub)
    .add_property("input_divide", make_function(&mach::LinearMachine::getInputDivision, return_value_policy<copy_const_reference>()), &set_input_div)
    .add_property("weights", make_function(&mach::LinearMachine::getWeights, return_value_policy<copy_const_reference>()), &set_weight)
    .add_property("biases", make_function(&mach::LinearMachine::getBiases, return_value_policy<copy_const_reference>()), &set_bias)
    .add_property("activation", &mach::LinearMachine::getActivation, &mach::LinearMachine::setActivation)
    .add_property("shape", &get_shape, &set_shape)
    .def("resize", &mach::LinearMachine::resize, (arg("self"), arg("input"), arg("output")), "Resizes the machine. If either the input or output increases in size, the weights and other factors should be considered uninitialized. If the size is preserved or reduced, already initialized values will not be changed.\n\nTip: Use this method to force data compression. All will work out given most relevant factors to be preserved are organized on the top of the weight matrix. In this way, reducing the system size will supress less relevant projections.")
    .def("__call__", &forward2, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("forward", &forward2, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("__call__", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("forward", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    ;
}
