/**
 * @file python/machine/src/plda.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings for the PLDA{Base,}Machine
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#include <boost/shared_ptr.hpp>
#include "core/python/exception.h"
#include "machine/PLDAMachine.h"

using namespace boost::python;
namespace mach = bob::machine;
namespace io = bob::io;
namespace ca = bob::core::array;
namespace tp = bob::python;

static double computeLikelihood1(mach::PLDAMachine& plda, 
  const blitz::Array<double, 1>& sample, bool with_enrolled_samples=true)
{
  return plda.computeLikelihood(sample, with_enrolled_samples);
}

static double computeLikelihood2(mach::PLDAMachine& plda, 
  const blitz::Array<double, 2>& samples, bool with_enrolled_samples=true)
{
  return plda.computeLikelihood(samples, with_enrolled_samples);
}

static double plda_forward_sample(mach::PLDAMachine& m, 
    tp::const_ndarray samples) {
  const ca::typeinfo& info = samples.type();

  if (info.dtype != ca::t_float64) 
    PYTHON_ERROR(TypeError, "PLDA forwarding does not accept type '%s'",
        info.str().c_str());

  switch (info.nd) {
    case 1:
      {
        double score;
        // Calls the forward function
        m.forward(samples.bz<double,1>(), score);
        return score;
      }
      break;
    case 2:
      {
        double score;
        // Calls the forward function
        m.forward(samples.bz<double,2>(), score);
        return score;
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "PLDA forwarding does not accept type '%s'",
          info.str().c_str());
  }
}

static object pldabase_getAddGamma(mach::PLDABaseMachine& m, const size_t a) {
  blitz::Array<double,2> res = m.getAddGamma(a).copy();
  return object(res);
}

static object plda_getAddGamma(mach::PLDAMachine& m, const size_t a) {
  blitz::Array<double,2> res = m.getAddGamma(a).copy();
  return object(res);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(computeLikelihood1_overloads, computeLikelihood1, 2, 3)
BOOST_PYTHON_FUNCTION_OVERLOADS(computeLikelihood2_overloads, computeLikelihood2, 2, 3)


void bind_machine_plda() {
  class_<mach::PLDABaseMachine, boost::shared_ptr<mach::PLDABaseMachine> >("PLDABaseMachine", "A PLDABaseMachine", init<const size_t, const size_t, const size_t>((arg("d"), arg("nf"), arg("ng")), "Builds a new PLDABaseMachine. A PLDABaseMachine can be seen as a container for F, G, sigma and mu when performing Probabilistic Linear Discriminant Analysis (PLDA)."))
    .def(init<>("Constructs a new empty PLDABaseMachine."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new PLDABaseMachine from a configuration file."))
    .def(init<const mach::PLDABaseMachine&>((arg("machine")), "Copy constructs a PLDABaseMachine"))
    .def("load", &mach::PLDABaseMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &mach::PLDABaseMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("F", make_function(&mach::PLDABaseMachine::getF, return_value_policy<copy_const_reference>()), &mach::PLDABaseMachine::setF)
    .add_property("G", make_function(&mach::PLDABaseMachine::getG, return_value_policy<copy_const_reference>()), &mach::PLDABaseMachine::setG)
    .add_property("mu", make_function(&mach::PLDABaseMachine::getMu, return_value_policy<copy_const_reference>()), &mach::PLDABaseMachine::setMu)
    .add_property("sigma", make_function(&mach::PLDABaseMachine::getSigma, return_value_policy<copy_const_reference>()), &mach::PLDABaseMachine::setSigma)
    .add_property("__isigma__", make_function(&mach::PLDABaseMachine::getISigma, return_value_policy<copy_const_reference>()))
    .add_property("__alpha__", make_function(&mach::PLDABaseMachine::getAlpha, return_value_policy<copy_const_reference>()))
    .add_property("__beta__", make_function(&mach::PLDABaseMachine::getBeta, return_value_policy<copy_const_reference>()))
    .add_property("__FtBeta__", make_function(&mach::PLDABaseMachine::getFtBeta, return_value_policy<copy_const_reference>()))
    .add_property("__GtISigma__", make_function(&mach::PLDABaseMachine::getGtISigma, return_value_policy<copy_const_reference>()))
    .add_property("__logdetAlpha__", &mach::PLDABaseMachine::getLogDetAlpha)
    .add_property("__logdetSigma__", &mach::PLDABaseMachine::getLogDetSigma)
    .def("hasGamma", &mach::PLDABaseMachine::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("computeGamma", &mach::PLDABaseMachine::computeGamma, (arg("self"), arg("a"), arg("gamma")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("getAddGamma", &pldabase_getAddGamma, (arg("self"), arg("a")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("hasLogLikeConstTerm", &mach::PLDABaseMachine::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("computeLogLikeConstTerm", (double (mach::PLDABaseMachine::*)(const size_t, const blitz::Array<double,2>&))&mach::PLDABaseMachine::computeLogLikeConstTerm, (arg("self"), arg("a"), arg("gamma")), "Computes the log likelihood constant term for the given number of samples.")
    .def("getAddLogLikeConstTerm", &mach::PLDABaseMachine::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .add_property("DimD", &mach::PLDABaseMachine::getDimD)
    .add_property("DimF", &mach::PLDABaseMachine::getDimF)
    .add_property("DimG", &mach::PLDABaseMachine::getDimG)
  ;

  class_<mach::PLDAMachine, boost::shared_ptr<mach::PLDAMachine> >("PLDAMachine", "A PLDAMachine", init<boost::shared_ptr<mach::PLDABaseMachine> >((arg("plda_base")), "Builds a new PLDAMachine. An attached PLDABaseMachine should be provided, containing the PLDA model (F, G and Sigma). The PLDAMachine only carries information the enrolled samples."))
    .def(init<>("Constructs a new empty PLDAMachine."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new PLDAMachine from a configuration file."))
    .def(init<const mach::PLDAMachine&>((arg("machine")), "Copy constructs a PLDAMachine"))
    .def("load", &mach::PLDAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &mach::PLDAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("plda_base", &mach::PLDAMachine::getPLDABase, &mach::PLDAMachine::setPLDABase)
    .add_property("DimD", &mach::PLDAMachine::getDimD)
    .add_property("DimF", &mach::PLDAMachine::getDimF)
    .add_property("DimG", &mach::PLDAMachine::getDimG)
    .add_property("n_samples", &mach::PLDAMachine::getNSamples, &mach::PLDAMachine::setNSamples)
    .add_property("WSumXitBetaXi", &mach::PLDAMachine::getWSumXitBetaXi, &mach::PLDAMachine::setWSumXitBetaXi)
    .add_property("weightedSum", make_function(&mach::PLDAMachine::getWeightedSum, return_value_policy<copy_const_reference>()), &mach::PLDAMachine::setWeightedSum)
    .add_property("log_likelihood", &mach::PLDAMachine::getLogLikelihood, &mach::PLDAMachine::setLogLikelihood)
    .def("hasGamma", &mach::PLDAMachine::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("getAddGamma", &plda_getAddGamma, (arg("self"), arg("a")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("hasLogLikeConstTerm", &mach::PLDAMachine::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("getAddLogLikeConstTerm", &mach::PLDAMachine::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .def("computeLikelihood", &computeLikelihood1, computeLikelihood1_overloads((arg("self"), arg("sample"), arg("use_enrolled_samples")=true), "Computes the likelihood considering only the probe sample or jointly the probe sample and the enrolled samples."))
    .def("computeLikelihood", &computeLikelihood2, computeLikelihood2_overloads((arg("self"), arg("samples"), arg("use_enrolled_samples")=true), "Computes the likelihood considering only the probe samples or jointly the probes samples and the enrolled samples."))
    .def("__call__", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a score.")
    .def("forward", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a score.")
  ;
}
