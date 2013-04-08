/**
 * @file machine/python/gaussian.cc
 * @date Tue Jul 26 15:11:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include <boost/concept_check.hpp>
#include <bob/machine/Gaussian.h>

#include <bob/core/python/ndarray.h>

using namespace boost::python;


static object py_getMean(const bob::machine::Gaussian& machine) {
  size_t n_inputs = machine.getNInputs();
  bob::python::ndarray mean(bob::core::array::t_float64, n_inputs);
  blitz::Array<double,1> mean_ = mean.bz<double,1>();
  mean_ = machine.getMean();
  return mean.self();
}

static void py_setMean(bob::machine::Gaussian& machine, bob::python::const_ndarray mean) {
  const bob::core::array::typeinfo& info = mean.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> mean_ = mean.bz<double,1>();
  machine.setMean(mean_);
}

static object py_getVariance(const bob::machine::Gaussian& machine) {
  size_t n_inputs = machine.getNInputs();
  bob::python::ndarray variance(bob::core::array::t_float64, n_inputs);
  blitz::Array<double,1> variance_ = variance.bz<double,1>();
  variance_ = machine.getVariance();
  return variance.self();
}

static void py_setVariance(bob::machine::Gaussian& machine, bob::python::const_ndarray variance) {
  const bob::core::array::typeinfo& info = variance.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> variance_ = variance.bz<double,1>();
  machine.setVariance(variance_);
}

static object py_getVarianceThresholds(const bob::machine::Gaussian& machine) {
  size_t n_inputs = machine.getNInputs();
  bob::python::ndarray varianceThresholds(bob::core::array::t_float64, n_inputs);
  blitz::Array<double,1> varianceThresholds_ = varianceThresholds.bz<double,1>();
  varianceThresholds_ = machine.getVarianceThresholds();
  return varianceThresholds.self();
}

static void py_setVarianceThresholds(bob::machine::Gaussian& machine, bob::python::const_ndarray varianceThresholds) {
  const bob::core::array::typeinfo& info = varianceThresholds.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> varianceThresholds_ = varianceThresholds.bz<double,1>();
  machine.setVarianceThresholds(varianceThresholds_);
}

static double py_logLikelihood(const bob::machine::Gaussian& machine, bob::python::const_ndarray input) {
  double output;
  machine.forward(input.bz<double,1>(), output);
  return output;
}

static double py_logLikelihood_(const bob::machine::Gaussian& machine, bob::python::const_ndarray input) {
  double output;
  machine.forward_(input.bz<double,1>(), output);
  return output;
}

void bind_machine_gaussian()
{
  class_<bob::machine::Gaussian, boost::shared_ptr<bob::machine::Gaussian>, bases<bob::machine::Machine<blitz::Array<double,1>, double> > >("Gaussian",
    "This class implements a multivariate diagonal Gaussian distribution.", init<>())
    .def(init<const size_t>(args("n_inputs")))
    .def(init<bob::machine::Gaussian&>(args("other")))
    .def(init<bob::io::HDF5File&>(args("config")))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::Gaussian::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this Gaussian with the 'other' one to be approximately the same.")
    .add_property("dim_d", &bob::machine::Gaussian::getNInputs, &bob::machine::Gaussian::setNInputs,
      "Dimensionality of the input feature space")
    .add_property("mean", &py_getMean, &py_setMean, "Mean of the Gaussian")
    .add_property("variance", &py_getVariance, &py_setVariance, "The diagonal of the (diagonal) covariance matrix")
    .add_property("variance_thresholds", &py_getVarianceThresholds, &py_setVarianceThresholds,
      "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. "
      "The variance will be set to this value if an attempt is made to set it to a smaller value.")
    .def("set_variance_thresholds",  (void (bob::machine::Gaussian::*)(const double))&bob::machine::Gaussian::setVarianceThresholds,
         "Set the variance flooring thresholds equal to the given threshold for all the dimensions.")
    .def("resize", &bob::machine::Gaussian::resize, "Set the input dimensionality, reset the mean to zero and the variance to one.")
    .def("log_likelihood", &py_logLikelihood, "Output the log likelihood of the sample, x. The input size is checked.")
    .def("log_likelihood_", &py_logLikelihood_, "Output the log likelihood of the sample, x. The input size is NOT checked.")
    .def("save", &bob::machine::Gaussian::save, "Save to a Configuration")
    .def("load", &bob::machine::Gaussian::load, "Load from a Configuration")
    .def(self_ns::str(self_ns::self))
  ;
}
