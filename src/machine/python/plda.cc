/**
 * @file machine/python/plda.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings for the PLDA{Base,}Machine
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

#include "bob/core/python/ndarray.h"
#include <boost/shared_ptr.hpp>
#include "bob/core/python/exception.h"
#include "bob/machine/PLDAMachine.h"

using namespace boost::python;


// Set and Get methods that uses blitz::Arrays
static object py_get_mu(const bob::machine::PLDABaseMachine& machine) 
{
  const size_t dim_d = machine.getDimD();
  bob::python::ndarray mu(bob::core::array::t_float64, dim_d);
  blitz::Array<double,1> mu_ = mu.bz<double,1>();
  mu_ = machine.getMu();
  return mu.self();
}

static void py_set_mu(bob::machine::PLDABaseMachine& machine, 
  bob::python::const_ndarray mu) 
{
  const blitz::Array<double,1> mu_ = mu.bz<double,1>();
  machine.setMu(mu_);
}

static object py_get_f(const bob::machine::PLDABaseMachine& machine) 
{
  const size_t dim_d = machine.getDimD();
  const size_t dim_f = machine.getDimF();
  bob::python::ndarray f(bob::core::array::t_float64, dim_d, dim_f);
  blitz::Array<double,2> f_ = f.bz<double,2>();
  f_ = machine.getF();
  return f.self();
}

static void py_set_f(bob::machine::PLDABaseMachine& machine, 
  bob::python::const_ndarray f) 
{
  const blitz::Array<double,2> f_ = f.bz<double,2>();
  machine.setF(f_);
}

static object py_get_g(const bob::machine::PLDABaseMachine& machine) 
{
  const size_t dim_d = machine.getDimD();
  const size_t dim_g = machine.getDimG();
  bob::python::ndarray g(bob::core::array::t_float64, dim_d, dim_g);
  blitz::Array<double,2> g_ = g.bz<double,2>();
  g_ = machine.getG();
  return g.self();
}

static void py_set_g(bob::machine::PLDABaseMachine& machine, 
  bob::python::const_ndarray g) 
{
  const blitz::Array<double,2> g_ = g.bz<double,2>();
  machine.setG(g_);
}

static object py_get_sigma(const bob::machine::PLDABaseMachine& machine) 
{
  const size_t dim_d = machine.getDimD();
  bob::python::ndarray sigma(bob::core::array::t_float64, dim_d);
  blitz::Array<double,1> sigma_ = sigma.bz<double,1>();
  sigma_ = machine.getSigma();
  return sigma.self();
}

static void py_set_sigma(bob::machine::PLDABaseMachine& machine, 
  bob::python::const_ndarray sigma) 
{
  const blitz::Array<double,1> sigma_ = sigma.bz<double,1>();
  machine.setSigma(sigma_);
}

static object py_get_variance_thresholds(const bob::machine::PLDABaseMachine& machine) 
{
  const size_t dim_d = machine.getDimD();
  bob::python::ndarray varianceThresholds(bob::core::array::t_float64, dim_d);
  blitz::Array<double,1> varianceThresholds_ = varianceThresholds.bz<double,1>();
  varianceThresholds_ = machine.getVarianceThresholds();
  return varianceThresholds.self();
}

static void py_set_variance_thresholds(bob::machine::PLDABaseMachine& machine, 
  bob::python::const_ndarray varianceThresholds) 
{
  const blitz::Array<double,1> varianceThresholds_ = varianceThresholds.bz<double,1>();
  machine.setVarianceThresholds(varianceThresholds_);
}


static double computeLogLikelihood1(bob::machine::PLDAMachine& plda, 
  const blitz::Array<double, 1>& sample, bool with_enrolled_samples=true)
{
  return plda.computeLogLikelihood(sample, with_enrolled_samples);
}

static double computeLogLikelihood2(bob::machine::PLDAMachine& plda, 
  const blitz::Array<double, 2>& samples, bool with_enrolled_samples=true)
{
  return plda.computeLogLikelihood(samples, with_enrolled_samples);
}

static double plda_forward_sample(bob::machine::PLDAMachine& m, 
    bob::python::const_ndarray samples) {
  const bob::core::array::typeinfo& info = samples.type();

  if (info.dtype != bob::core::array::t_float64) 
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

static double py_log_likelihood_point_estimate(bob::machine::PLDABaseMachine& plda, 
  bob::python::const_ndarray xij, bob::python::const_ndarray hi, 
  bob::python::const_ndarray wij)
{
  const blitz::Array<double,1> xij_ = xij.bz<double,1>();
  const blitz::Array<double,1> hi_ = hi.bz<double,1>();
  const blitz::Array<double,1> wij_ = wij.bz<double,1>();
  return plda.computeLogLikelihoodPointEstimate(xij_, hi_, wij_);
}

static object pldabase_getAddGamma(bob::machine::PLDABaseMachine& m, const size_t a) {
  blitz::Array<double,2> res = m.getAddGamma(a).copy();
  return object(res);
}

static object pldabase_getGamma(bob::machine::PLDABaseMachine& m, const size_t a) {
  blitz::Array<double,2> res = m.getGamma(a).copy();
  return object(res);
}

static object plda_getAddGamma(bob::machine::PLDAMachine& m, const size_t a) {
  blitz::Array<double,2> res = m.getAddGamma(a).copy();
  return object(res);
}

static object plda_getGamma(bob::machine::PLDAMachine& m, const size_t a) {
  blitz::Array<double,2> res = m.getGamma(a).copy();
  return object(res);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(computeLogLikelihood1_overloads, computeLogLikelihood1, 2, 3)
BOOST_PYTHON_FUNCTION_OVERLOADS(computeLogLikelihood2_overloads, computeLogLikelihood2, 2, 3)

void bind_machine_plda() 
{
  class_<bob::machine::PLDABaseMachine, boost::shared_ptr<bob::machine::PLDABaseMachine> >("PLDABaseMachine", "A PLDABaseMachine can be seen as a container for F, G, sigma and mu when performing Probabilistic Linear Discriminant Analysis (PLDA).\nReferences:\n1. 'Probabilistic Linear Discriminant Analysis for Inference About Identity', Prince and Elder, ICCV'2007.\n2. 'Probabilistic Models for Inference about Identity', Li, Fu, Mohammed, Elder and Prince, PAMI'2012.", init<const size_t, const size_t, const size_t, optional<const double> >((arg("dim_d"), arg("dim_f"), arg("dim_g"), arg("variance_flooring")=0.), "Builds a new PLDABaseMachine. dim_d is the dimensionality of the input features, dim_f is the dimensionality of the F subspace and dim_g the dimensionality of the G subspace. The variance flooring threshold is the minimum value that the variance sigma can reach, as this diagonal matrix is inverted."))
    .def(init<>("Constructs a new empty PLDABaseMachine."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new PLDABaseMachine from a configuration file."))
    .def(init<const bob::machine::PLDABaseMachine&>((arg("machine")), "Copy constructs a PLDABaseMachine"))
    .def(self == self)
    .def(self != self)
    .def("load", &bob::machine::PLDABaseMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::PLDABaseMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("dim_d", &bob::machine::PLDABaseMachine::getDimD, "Dimensionality of the input feature vectors")
    .add_property("dim_f", &bob::machine::PLDABaseMachine::getDimF, "Dimensionality of the F subspace/matrix of the PLDA model")
    .add_property("dim_g", &bob::machine::PLDABaseMachine::getDimG, "Dimensionality of the G subspace/matrix of the PLDA model")
    .add_property("mu", &py_get_mu, &py_set_mu, "The mean vector mu of the PLDA model")
    .add_property("f", &py_get_f, &py_set_f, "The subspace/matrix F of the PLDA model")
    .add_property("g", &py_get_g, &py_set_g, "The subspace/matrix G of the PLDA model")
    .add_property("sigma", &py_get_sigma, &py_set_sigma, "The diagonal covariance matrix (represented by a 1D numpy array) sigma of the PLDA model")
    .add_property("variance_thresholds", &py_get_variance_thresholds, &py_set_variance_thresholds,
      "The variance flooring thresholds, i.e. the minimum allowed value of variance (sigma) in each dimension. "
      "The variance sigma will be set to this value if an attempt is made to set it to a smaller value.")
    .def("resize", &bob::machine::PLDABaseMachine::resize, (arg("self"), arg("dim_d"), arg("dim_f"), arg("dim_g"), arg("variance_threshold")=0.), "Resizes the dimensionality of the PLDA model. Paramaters mu, F, G and sigma are reinitialized (variance flooring thresholds as well).")
    .def("has_gamma", &bob::machine::PLDABaseMachine::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("compute_gamma", &bob::machine::PLDABaseMachine::computeGamma, (arg("self"), arg("a"), arg("gamma")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_add_gamma", &pldabase_getAddGamma, (arg("self"), arg("a")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_gamma", &pldabase_getGamma, (arg("self"), arg("a")), "Returns the gamma matrix for the given number of samples if it has already been put in cache. Throws an exception otherwise. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("has_log_like_const_term", &bob::machine::PLDABaseMachine::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("compute_log_like_const_term", (double (bob::machine::PLDABaseMachine::*)(const size_t, const blitz::Array<double,2>&) const)&bob::machine::PLDABaseMachine::computeLogLikeConstTerm, (arg("self"), arg("a"), arg("gamma")), "Computes the log likelihood constant term for the given number of samples.")
    .def("get_add_log_like_const_term", &bob::machine::PLDABaseMachine::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .def("get_log_like_const_term", &bob::machine::PLDABaseMachine::getLogLikeConstTerm, (arg("self"), arg("a")), "Returns the log likelihood constant term for the given number of samples if it has already been put in cache. Throws an exception otherwise.")
    .def("clear_maps", &bob::machine::PLDABaseMachine::clearMaps, (arg("self")), "Clear the maps containing the gamma's as well as the log likelihood constant term for few number of samples. These maps are used to make likelihood computations faster.")
    .def("compute_log_likelihood_point_estimate", &py_log_likelihood_point_estimate, (arg("self"), arg("xij"), arg("hi"), arg("wij")), "Computes the log-likelihood of a sample given the latent variables hi and wij (point estimate rather than Bayesian-like full integration).")
    .def(self_ns::str(self_ns::self))
    .add_property("__isigma__", make_function(&bob::machine::PLDABaseMachine::getISigma, return_value_policy<copy_const_reference>()), "sigma^{-1} matrix stored in cache")
    .add_property("__alpha__", make_function(&bob::machine::PLDABaseMachine::getAlpha, return_value_policy<copy_const_reference>()), "alpha matrix stored in cache")
    .add_property("__beta__", make_function(&bob::machine::PLDABaseMachine::getBeta, return_value_policy<copy_const_reference>()), "beta matrix stored in cache")
    .add_property("__ft_beta__", make_function(&bob::machine::PLDABaseMachine::getFtBeta, return_value_policy<copy_const_reference>()), "F^T.beta matrix stored in cache")
    .add_property("__gt_i_sigma__", make_function(&bob::machine::PLDABaseMachine::getGtISigma, return_value_policy<copy_const_reference>()), "G^T.sigma^{-1} matrix stored in cache")
    .add_property("__logdet_alpha__", &bob::machine::PLDABaseMachine::getLogDetAlpha, "Logarithm of the determinant of the alpha matrix stored in cache.")
    .add_property("__logdet_sigma__", &bob::machine::PLDABaseMachine::getLogDetSigma, "Logarithm of the determinant of the sigma matrix stored in cache.")
    .def("__precompute__", &bob::machine::PLDABaseMachine::precompute, (arg("self")), "Precomputes useful values such as alpha and beta.")
    .def("__precompute_log_like__", &bob::machine::PLDABaseMachine::precomputeLogLike, (arg("self")), "Precomputes useful values for log-likelihood computations.")
  ;

  class_<bob::machine::PLDAMachine, boost::shared_ptr<bob::machine::PLDAMachine> >("PLDAMachine", "A PLDAMachine", init<boost::shared_ptr<bob::machine::PLDABaseMachine> >((arg("plda_base")), "Builds a new PLDAMachine. An attached PLDABaseMachine should be provided, containing the PLDA model (F, G and Sigma). The PLDAMachine only carries information the enrolled samples."))
    .def(init<>("Constructs a new empty PLDAMachine."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new PLDAMachine from a configuration file."))
    .def(init<const bob::machine::PLDAMachine&>((arg("machine")), "Copy constructs a PLDAMachine"))
    .def(self == self)
    .def(self != self)
    .def("load", &bob::machine::PLDAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::PLDAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("plda_base", &bob::machine::PLDAMachine::getPLDABase, &bob::machine::PLDAMachine::setPLDABase)
    .add_property("dim_d", &bob::machine::PLDAMachine::getDimD, "Dimensionality of the input feature vectors")
    .add_property("dim_f", &bob::machine::PLDAMachine::getDimF, "Dimensionality of the F subspace/matrix of the PLDA model")
    .add_property("dim_g", &bob::machine::PLDAMachine::getDimG, "Dimensionality of the G subspace/matrix of the PLDA model")
    .add_property("n_samples", &bob::machine::PLDAMachine::getNSamples, &bob::machine::PLDAMachine::setNSamples)
    .add_property("w_sum_xit_beta_xi", &bob::machine::PLDAMachine::getWSumXitBetaXi, &bob::machine::PLDAMachine::setWSumXitBetaXi)
    .add_property("weighted_sum", make_function(&bob::machine::PLDAMachine::getWeightedSum, return_value_policy<copy_const_reference>()), &bob::machine::PLDAMachine::setWeightedSum)
    .add_property("log_likelihood", &bob::machine::PLDAMachine::getLogLikelihood, &bob::machine::PLDAMachine::setLogLikelihood)
    .def("has_gamma", &bob::machine::PLDAMachine::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_add_gamma", &plda_getAddGamma, (arg("self"), arg("a")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_gamma", &plda_getGamma, (arg("self"), arg("a")), "Returns the gamma matrix for the given number of samples if it has already been put in cache. Throws an exception otherwise. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("has_log_like_const_term", &bob::machine::PLDAMachine::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("get_add_log_like_const_term", &bob::machine::PLDAMachine::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .def("get_log_like_const_term", &bob::machine::PLDAMachine::getLogLikeConstTerm, (arg("self"), arg("a")), "Returns the log likelihood constant term for the given number of samples if it has already been put in cache. Throws an exception otherwise.")
    .def("clear_maps", &bob::machine::PLDAMachine::clearMaps, (arg("self")), "Clear the maps containing the gamma's as well as the log likelihood constant term for few number of samples. These maps are used to make likelihood computations faster.")
    .def("compute_log_likelihood", &computeLogLikelihood1, computeLogLikelihood1_overloads((arg("self"), arg("sample"), arg("use_enrolled_samples")=true), "Computes the likelihood considering only the probe sample or jointly the probe sample and the enrolled samples."))
    .def("compute_log_likelihood", &computeLogLikelihood2, computeLogLikelihood2_overloads((arg("self"), arg("samples"), arg("use_enrolled_samples")=true), "Computes the likelihood considering only the probe samples or jointly the probes samples and the enrolled samples."))
    .def("__call__", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a score.")
    .def("forward", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a score.")
  ;
}
