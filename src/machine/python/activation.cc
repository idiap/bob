/**
 * @file machine/python/activation.cc
 * @date Thu Jul 7 16:49:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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

#include <boost/python.hpp>
#include <bob/machine/Activation.h>
#include <bob/machine/ActivationRegistry.h>

using namespace boost::python;

static bool activation_is_equal(boost::shared_ptr<bob::machine::Activation> a,
    boost::shared_ptr<bob::machine::Activation> b) {
  return a->str() == b->str();
}

void bind_machine_activation() {
  class_<bob::machine::Activation, boost::shared_ptr<bob::machine::Activation>, boost::noncopyable>("Activation", 
      "Base class for activation functions", no_init)
    .def("f", &bob::machine::Activation::f, (arg("self"), arg("z")),
        "Computes activated value, given an input ``z``")
    .def("__call__", &bob::machine::Activation::f, (arg("self"), arg("z")), 
        "Computes activated value, given an input ``z``")
    .def("f_prime", &bob::machine::Activation::f_prime, 
        (arg("z")), "Computes the derivative of the activated value.")
    .def("f_prime_from_f", &bob::machine::Activation::f_prime_from_f, 
        (arg("z")), "Computes the derivative of the activated value, given **the activation used to compute the activated value originally**.")
    .def("save", &bob::machine::Activation::save, (arg("self"), arg("h5f")), 
       "Saves itself to a :py:class:`bob.io.HDF5File`")
    .def("load", &bob::machine::Activation::load, (arg("self"), arg("h5f")), 
       "Loads itself from a :py:class:`bob.io.HDF5File`")
    .def("unique_identifier", 
        &bob::machine::Activation::unique_identifier, (arg("self")),
        "Returns a unique identifier, used by this class in connection to the Activation registry.")
    .def("__str__", &bob::machine::Activation::str)
    .def("__eq__", &activation_is_equal)
    ;

  class_<bob::machine::IdentityActivation, boost::shared_ptr<bob::machine::IdentityActivation>, bases<bob::machine::Activation> >("IdentityActivation", "Computes :math:`f(z) = z` as activation function", init<>((arg("self"))))
    ;

  class_<bob::machine::LinearActivation, boost::shared_ptr<bob::machine::LinearActivation>, bases<bob::machine::Activation> >("LinearActivation", "Computes :math:`f(z) = C \\cdot z` as activation function", init<optional<double> >((arg("self"), arg("C")=1.), "Builds a new linear activation function with a given constant. Don't use this if you just want to set constant to the default value (1.0). In such a case, prefer to use the more efficient :py:class:`bob.machine.IdentityActivation`."))
    .add_property("C", &bob::machine::LinearActivation::C, "The multiplication factor for the linear function")

    ;
  class_<bob::machine::HyperbolicTangentActivation, boost::shared_ptr<bob::machine::HyperbolicTangentActivation>, bases<bob::machine::Activation> >("HyperbolicTangentActivation", "Computes :math:`f(z) = \\tanh(z)` as activation function", init<>((arg("self"))))
    ;

  class_<bob::machine::MultipliedHyperbolicTangentActivation, boost::shared_ptr<bob::machine::MultipliedHyperbolicTangentActivation>, bases<bob::machine::Activation> >("MultipliedHyperbolicTangentActivation", "Computes :math:`f(z) = C \\cdot \\tanh(Mz)` as activation function", init<optional<double, double> >((arg("self"), arg("C")=1., arg("M")=1.), "Builds a new hyperbolic tangent activation fucntion with a given constant for the inner and outter products. Don't use this if you just want to set the constants to the default values (1.0). In such a case, prefer to use the more efficient :py:class:`bob.machine.HyperbolicTangentActivation`."))
    .add_property("C", &bob::machine::MultipliedHyperbolicTangentActivation::C, "The outside multiplication factor for the hyperbolic tangent function")
    .add_property("M", &bob::machine::MultipliedHyperbolicTangentActivation::M, "The inner multiplication factor for the argument")
    ;

  class_<bob::machine::LogisticActivation, boost::shared_ptr<bob::machine::LogisticActivation>, bases<bob::machine::Activation> >("LogisticActivation", "Computes the so-called logistic function: :math:`f(z) = 1. / (1. + e^{-z})` as activation function", init<>((arg("self"))))
    ;
}
