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
#include <bob/python/ndarray.h>
#include <bob/machine/Gaussian.h>


using namespace boost::python;

static void py_setMean(bob::machine::Gaussian& machine, 
  bob::python::const_ndarray mean)
{
  machine.setMean(mean.bz<double,1>());
}

static void py_setVariance(bob::machine::Gaussian& machine,
  bob::python::const_ndarray variance)
{
  machine.setVariance(variance.bz<double,1>());
}

static void py_setVarianceThresholds(bob::machine::Gaussian& machine, 
  bob::python::const_ndarray varianceThresholds) 
{
  machine.setVarianceThresholds(varianceThresholds.bz<double,1>());
}

static tuple get_shape(const bob::machine::Gaussian& m)
{
  return make_tuple(m.getNInputs());
}

static void set_shape(bob::machine::Gaussian& m,
  const blitz::TinyVector<int,1>& s)
{
  m.resize(s(0));
}

static double py_logLikelihood(const bob::machine::Gaussian& machine,
  bob::python::const_ndarray input)
{
  double output;
  machine.forward(input.bz<double,1>(), output);
  return output;
}

static double py_logLikelihood_(const bob::machine::Gaussian& machine,
  bob::python::const_ndarray input)
{
  double output;
  machine.forward_(input.bz<double,1>(), output);
  return output;
}

void bind_machine_gaussian()
{
  class_<bob::machine::Gaussian, boost::shared_ptr<bob::machine::Gaussian>, bases<bob::machine::Machine<blitz::Array<double,1>, double> > >("Gaussian",
    "This class implements a multivariate diagonal Gaussian distribution.", init<>(arg("self")))
    .def(init<const size_t>((arg("self"), arg("n_inputs"))))
    .def(init<bob::machine::Gaussian&>((arg("self"), arg("other"))))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("config"))))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::Gaussian::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this Gaussian with the 'other' one to be approximately the same.")
    .add_property("dim_d", &bob::machine::Gaussian::getNInputs, &bob::machine::Gaussian::setNInputs,
      "Dimensionality of the input feature space")
    .add_property("mean", make_function(&bob::machine::Gaussian::getMean, return_value_policy<copy_const_reference>()), &py_setMean, "Mean of the Gaussian")
    .add_property("variance", make_function(&bob::machine::Gaussian::getVariance, return_value_policy<copy_const_reference>()), &py_setVariance, "The diagonal of the (diagonal) covariance matrix")
    .add_property("variance_thresholds", make_function(&bob::machine::Gaussian::getVarianceThresholds, return_value_policy<copy_const_reference>()), &py_setVarianceThresholds,
      "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. "
      "The variance will be set to this value if an attempt is made to set it to a smaller value.")
    .add_property("shape", &get_shape, &set_shape, "A tuple that represents the dimensionality of the Gaussian ``(dim_d,)``.")
    .def("set_variance_thresholds",  (void (bob::machine::Gaussian::*)(const double))&bob::machine::Gaussian::setVarianceThresholds, (arg("self"), arg("var_thd")),
         "Set the variance flooring thresholds equal to the given threshold for all the dimensions.")
    .def("resize", &bob::machine::Gaussian::resize, (arg("self"), arg("dim_d")), "Set the input dimensionality, reset the mean to zero and the variance to one.")
    .def("log_likelihood", &py_logLikelihood, (arg("self"), arg("sample")), "Output the log likelihood of the sample, x. The input size is checked.")
    .def("log_likelihood_", &py_logLikelihood_, (arg("self"), arg("sample")), "Output the log likelihood of the sample, x. The input size is NOT checked.")
    .def("save", &bob::machine::Gaussian::save, (arg("self"), arg("config")), "Save to a Configuration")
    .def("load", &bob::machine::Gaussian::load, (arg("self"), arg("config")),"Load from a Configuration")
    .def(self_ns::str(self_ns::self))
  ;
}
