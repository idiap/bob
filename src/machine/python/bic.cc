/**
 * @file machine/python/bic.cc
 * @date Wed Jun  6 10:29:09 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
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
#include <boost/python.hpp>
#include <bob/machine/BICMachine.h>
#include <bob/io/HDF5File.h>
#include <bob/core/python/exception.h>


static void bic_forward_(const bob::machine::BICMachine& machine, bob::python::const_ndarray input, bob::python::ndarray output){
  blitz::Array<double,1> o = output.bz<double,1>();
  machine.forward_(input.bz<double,1>(), o);
}

static void bic_forward(const bob::machine::BICMachine& machine, bob::python::const_ndarray input, bob::python::ndarray output){
  blitz::Array<double,1> o = output.bz<double,1>();
  machine.forward(input.bz<double,1>(), o);
}

static double bic_call(const bob::machine::BICMachine& machine, bob::python::const_ndarray input){
  blitz::Array<double,1> o(1);
  machine.forward(input.bz<double,1>(), o);
  return o(0);
}

void bind_machine_bic(){

  // bind exception
  bob::python::register_exception_translator<bob::machine::ZeroEigenvalueException>(PyExc_ZeroDivisionError);

  // bind BICMachine
  boost::python::class_<bob::machine::BICMachine, boost::shared_ptr<bob::machine::BICMachine> > (
      "BICMachine",
      "This machine is designed to classify image differences to be either intrapersonal or extrapersonal. "
      "There are two possible implementations of the BIC:\n"
      "\n"
      "* 'The Bayesian Intrapersonal/Extrapersonal Classifier' from Teixeira [1]_. "
      "  A full projection of the data is performed. No prior for the classes has to be selected.\n"
      "* 'Face Detection and Recognition using Maximum Likelihood Classifiers on Gabor Graphs' from Guenther and Wuertz [2]_."
      "  Only mean and variance of the difference vectors are calculated. There is no subspace truncation and no priors.\n"
      "\n"
      "What kind of machine is used is dependent on the way, this class is trained via the BICTrainer.\n"
      "\n"
      ".. [1] Marcio Luis Teixeira. The Bayesian intrapersonal/extrapersonal classifier. Colorado State University, 2003.\n"
      ".. [2] Manuel Guenther and Rolf P. Wuertz. Face detection and recognition using maximum likelihood classifiers on Gabor graphs. International Journal of Pattern Recognition and Artificial Intelligence, 23(3):433-461, 2009.",
      boost::python::init<bool>(
          (boost::python::arg("use_dffs") = false),
          "Initializes an empty BICMachine. The optional boolean parameter specifies whether to use the DFFS in the BIC implementation. \n\n.. warning :: Use this flag with care, the default value 'False' is usually the best choice!"
      )
    )

    .def(
      boost::python::init<const bob::machine::BICMachine&>(
          "Constructs one BICMachine from another one by doing a deep copy."
      )
    )

    .def(
      boost::python::self == boost::python::self
    )

    .def(
      "is_similar_to",
      &bob::machine::BICMachine::is_similar_to,
      (boost::python::arg("self"), boost::python::arg("other"), boost::python::arg("r_epsilon") = 1e-5, boost::python::arg("a_epsilon") = 1e-8),
      "Compares this BICMachine with the 'other' one to be approximately the same."
    )

    .def(
      "load",
      &bob::machine::BICMachine::load,
      (boost::python::arg("self"), boost::python::arg("file")),
      "Loads the configuration parameters from an hdf5 file."
    )

    .def(
      "save",
      &bob::machine::BICMachine::save,
      (boost::python::arg("self"), boost::python::arg("file")),
      "Saves the configuration parameters to an hdf5 file."
    )


    .def(
      "__call__",
      &bic_forward_,
      (
          boost::python::arg("self"),
          boost::python::arg("input"),
          boost::python::arg("output")
      ),
      "Computes the BIC or IEC score for the given input vector, which results of a comparison of two (facial) images. "
      "The score itself is the log-likelihood score of the given input vector belonging to the intrapersonal class. "
      "No sanity checks of input and output are performed."
    )

    .def(
      "__call__",
      &bic_call,
      (
          boost::python::arg("self"),
          boost::python::arg("input")
      ),
      "Computes the BIC or IEC score for the given input vector, which results of a comparison of two (facial) images. "
      "The resulting value is returned as a single float value. "
      "The score itself is the log-likelihood score of the given input vector belonging to the intrapersonal class. "
      "No sanity checks of input and output are performed."
    )

    .def(
      "forward_",
      &bic_forward_,
      (
          boost::python::arg("self"),
          boost::python::arg("input"),
          boost::python::arg("output")
      ),
      "Computes the BIC or IEC score for the given input vector, which results of a comparison of two (facial) images. "
      "The score itself is the log-likelihood score of the given input vector belonging to the intrapersonal class. "
      "No sanity checks of input and output are performed."
    )

    .def(
      "forward",
      &bic_forward,
      (
          boost::python::arg("self"),
          boost::python::arg("input"),
          boost::python::arg("output")
      ),
      "Computes the BIC or IEC score for the given input vector, which results of a comparison of two (facial) images. "
      "The score itself is the log-likelihood score of the given input vector belonging to the intrapersonal class. "
      "Sanity checks of input and output shape are performed."
    )

    .add_property(
      "use_dffs",
      // cast overloaded function with the same name to its type...
      static_cast<bool (bob::machine::BICMachine::*)() const>(&bob::machine::BICMachine::use_DFFS),
      static_cast<void (bob::machine::BICMachine::*)(bool)>(&bob::machine::BICMachine::use_DFFS),
      "Should the Distance From Feature Space (DFFS) measure be added during scoring? \n\n.. warning :: Only set this flag to True if the number of intrapersonal and extrapersonal training pairs is approximately equal. Otherwise, weird thing may happen!"
  );
}
