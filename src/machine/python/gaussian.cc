/**
 * @file machine/python/gaussian.cc
 * @date Tue Jul 26 15:11:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include <boost/python.hpp>
#include <boost/concept_check.hpp>
#include "bob/machine/Gaussian.h"

#include "bob/core/python/ndarray.h"

using namespace boost::python;
namespace io = bob::io;
namespace mach = bob::machine;
namespace bp = bob::python;
namespace ca = bob::core::array;


static object py_getMean(const mach::Gaussian& machine) {
  size_t n_inputs = machine.getNInputs();
  bp::ndarray mean(ca::t_float64, n_inputs);
  blitz::Array<double,1> mean_ = mean.bz<double,1>();
  mean_ = machine.getMean();
  return mean.self();
}

static void py_setMean(mach::Gaussian& machine, bp::const_ndarray mean) {
  const ca::typeinfo& info = mean.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> mean_ = mean.bz<double,1>();
  machine.setMean(mean_);
}

static object py_getVariance(const mach::Gaussian& machine) {
  size_t n_inputs = machine.getNInputs();
  bp::ndarray variance(ca::t_float64, n_inputs);
  blitz::Array<double,1> variance_ = variance.bz<double,1>();
  variance_ = machine.getVariance();
  return variance.self();
}

static void py_setVariance(mach::Gaussian& machine, bp::const_ndarray variance) {
  const ca::typeinfo& info = variance.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> variance_ = variance.bz<double,1>();
  machine.setVariance(variance_);
}

static object py_getVarianceThresholds(const mach::Gaussian& machine) {
  size_t n_inputs = machine.getNInputs();
  bp::ndarray varianceThresholds(ca::t_float64, n_inputs);
  blitz::Array<double,1> varianceThresholds_ = varianceThresholds.bz<double,1>();
  varianceThresholds_ = machine.getVarianceThresholds();
  return varianceThresholds.self();
}

static void py_setVarianceThresholds(mach::Gaussian& machine, bp::const_ndarray varianceThresholds) {
  const ca::typeinfo& info = varianceThresholds.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> varianceThresholds_ = varianceThresholds.bz<double,1>();
  machine.setVarianceThresholds(varianceThresholds_);
}

static double py_logLikelihood(const mach::Gaussian& machine, bp::const_ndarray input) {
  double output;
  machine.forward(input.bz<double,1>(), output);
  return output;
}

static double py_logLikelihood_(const mach::Gaussian& machine, bp::const_ndarray input) {
  double output;
  machine.forward_(input.bz<double,1>(), output);
  return output;
}

void bind_machine_gaussian() 
{  
  class_<mach::Gaussian, boost::shared_ptr<mach::Gaussian>, bases<mach::Machine<blitz::Array<double,1>, double> > >("Gaussian",
    "This class implements a multivariate diagonal Gaussian distribution.", init<>())
    .def(init<const size_t>(args("n_inputs")))
    .def(init<mach::Gaussian&>(args("other")))
    .def(init<io::HDF5File&>(args("config")))
    .def(self == self)
    .add_property("dim_d", &mach::Gaussian::getNInputs, &mach::Gaussian::setNInputs,
      "Dimensionality of the input feature space")
    .add_property("mean", &py_getMean, &py_setMean, "Mean of the Gaussian")
    .add_property("variance", &py_getVariance, &py_setVariance, "The diagonal of the (diagonal) covariance matrix")
    .add_property("variance_thresholds", &py_getVarianceThresholds, &py_setVarianceThresholds,
      "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. "
      "The variance will be set to this value if an attempt is made to set it to a smaller value.")
    .def("set_variance_thresholds",  (void (mach::Gaussian::*)(const double))&mach::Gaussian::setVarianceThresholds,
         "Set the variance flooring thresholds equal to the given threshold for all the dimensions.")
    .def("resize", &mach::Gaussian::resize, "Set the input dimensionality, reset the mean to zero and the variance to one.")
    .def("log_likelihood", &py_logLikelihood, "Output the log likelihood of the sample, x. The input size is checked.")
    .def("log_likelihood_", &py_logLikelihood_, "Output the log likelihood of the sample, x. The input size is NOT checked.")
    .def("save", &mach::Gaussian::save, "Save to a Configuration")
    .def("load", &mach::Gaussian::load, "Load from a Configuration")
    .def(self_ns::str(self_ns::self))
  ;
}
