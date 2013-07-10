/**
 * @file machine/python/plda.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings for the PLDABase/PLDAMachine
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

#include <bob/python/ndarray.h>
#include <boost/shared_ptr.hpp>
#include <bob/python/exception.h>
#include <bob/machine/PLDAMachine.h>

using namespace boost::python;

static void py_set_dim_d(bob::machine::PLDABase& machine, const size_t dim_d)
{
  machine.resize(dim_d, machine.getDimF(), machine.getDimG());
}
static void py_set_dim_f(bob::machine::PLDABase& machine, const size_t dim_f)
{
  machine.resize(machine.getDimD(), dim_f, machine.getDimG());
}
static void py_set_dim_g(bob::machine::PLDABase& machine, const size_t dim_g)
{
  machine.resize(machine.getDimD(), machine.getDimF(), dim_g);
}

// Set methods that uses blitz::Arrays
static void py_set_mu(bob::machine::PLDABase& machine,
  bob::python::const_ndarray mu)
{
  machine.setMu(mu.bz<double,1>());
}

static void py_set_f(bob::machine::PLDABase& machine,
  bob::python::const_ndarray f)
{
  machine.setF(f.bz<double,2>());
}

static void py_set_g(bob::machine::PLDABase& machine,
  bob::python::const_ndarray g)
{
  machine.setG(g.bz<double,2>());
}

static void py_set_sigma(bob::machine::PLDABase& machine,
  bob::python::const_ndarray sigma)
{
  machine.setSigma(sigma.bz<double,1>());
}


static double computeLogLikelihood(bob::machine::PLDAMachine& plda,
  bob::python::const_ndarray samples, bool with_enrolled_samples=true)
{
  const bob::core::array::typeinfo& info = samples.type();
  switch (info.nd) {
    case 1:
      return plda.computeLogLikelihood(samples.bz<double,1>(), with_enrolled_samples);
    case 2:
      return plda.computeLogLikelihood(samples.bz<double,2>(), with_enrolled_samples);
    default:
      PYTHON_ERROR(TypeError, "PLDA log-likelihood computation does not accept input array with '" SIZE_T_FMT "' dimensions (only 1D or 2D arrays)", info.nd);
  }
}

static double plda_forward_sample(bob::machine::PLDAMachine& m,
  bob::python::const_ndarray samples)
{
  const bob::core::array::typeinfo& info = samples.type();
  switch (info.nd) {
    case 1:
      {
        double score;
        // Calls the forward function
        m.forward(samples.bz<double,1>(), score);
        return score;
      }
    case 2:
      {
        double score;
        // Calls the forward function
        m.forward(samples.bz<double,2>(), score);
        return score;
      }
    default:
      PYTHON_ERROR(TypeError, "PLDA forwarding does not accept input array with '" SIZE_T_FMT "' dimensions (only 1D or 2D arrays)", info.nd);
  }
}

static double py_log_likelihood_point_estimate(bob::machine::PLDABase& plda,
  bob::python::const_ndarray xij, bob::python::const_ndarray hi,
  bob::python::const_ndarray wij)
{
  return plda.computeLogLikelihoodPointEstimate(xij.bz<double,1>(), 
           hi.bz<double,1>(), wij.bz<double,1>());
}

BOOST_PYTHON_FUNCTION_OVERLOADS(computeLogLikelihood_overloads, computeLogLikelihood, 2, 3)

void bind_machine_plda()
{
  class_<bob::machine::PLDABase, boost::shared_ptr<bob::machine::PLDABase> >("PLDABase", "A PLDABase can be seen as a container for the subspaces F, G, the diagonal covariance matrix sigma (stored as a 1D array) and the mean vector mu when performing Probabilistic Linear Discriminant Analysis (PLDA). PLDA is a probabilistic model that incorporates components describing both between-class and within-class variations. A PLDABase can be shared between several PLDAMachine that contains class-specific information (information about the enrolment samples).\n\nReferences:\n1. 'A Scalable Formulation of Probabilistic Linear Discriminant Analysis: Applied to Face Recognition', Laurent El Shafey, Chris McCool, Roy Wallace, Sebastien Marcel, TPAMI'2013\n2. 'Probabilistic Linear Discriminant Analysis for Inference About Identity', Prince and Elder, ICCV'2007.\n3. 'Probabilistic Models for Inference about Identity', Li, Fu, Mohammed, Elder and Prince, TPAMI'2012.", init<const size_t, const size_t, const size_t, optional<const double> >((arg("self"), arg("dim_d"), arg("dim_f"), arg("dim_g"), arg("variance_flooring")=0.), "Builds a new PLDABase. dim_d is the dimensionality of the input features, dim_f is the dimensionality of the F subspace and dim_g the dimensionality of the G subspace. The variance flooring threshold is the minimum value that the variance sigma can reach, as this diagonal matrix is inverted."))
    .def(init<>((arg("self")), "Constructs a new empty PLDABase."))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("config")), "Constructs a new PLDABase from a configuration file."))
    .def(init<const bob::machine::PLDABase&>((arg("self"), arg("machine")), "Copy constructs a PLDABase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::PLDABase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this PLDABase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::PLDABase::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::PLDABase::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("dim_d", &bob::machine::PLDABase::getDimD, &py_set_dim_d, "Dimensionality of the input feature vectors")
    .add_property("dim_f", &bob::machine::PLDABase::getDimF, &py_set_dim_f, "Dimensionality of the F subspace/matrix of the PLDA model")
    .add_property("dim_g", &bob::machine::PLDABase::getDimG, &py_set_dim_g, "Dimensionality of the G subspace/matrix of the PLDA model")
    .add_property("mu", make_function(&bob::machine::PLDABase::getMu, return_value_policy<copy_const_reference>()), &py_set_mu, "The mean vector mu of the PLDA model")
    .add_property("f", make_function(&bob::machine::PLDABase::getF, return_value_policy<copy_const_reference>()), &py_set_f, "The subspace/matrix F of the PLDA model")
    .add_property("g", make_function(&bob::machine::PLDABase::getG, return_value_policy<copy_const_reference>()), &py_set_g, "The subspace/matrix G of the PLDA model")
    .add_property("sigma", make_function(&bob::machine::PLDABase::getSigma, return_value_policy<copy_const_reference>()), &py_set_sigma, "The diagonal covariance matrix (represented by a 1D numpy array) sigma of the PLDA model")
    .add_property("variance_threshold", &bob::machine::PLDABase::getVarianceThreshold, &bob::machine::PLDABase::setVarianceThreshold,
      "The variance flooring threshold, i.e. the minimum allowed value of variance (sigma) in each dimension. "
      "The variance sigma will be set to this value if an attempt is made to set it to a smaller value.")
    .def("resize", &bob::machine::PLDABase::resize, (arg("self"), arg("dim_d"), arg("dim_f"), arg("dim_g")), "Resizes the dimensionality of the PLDA model. Paramaters mu, F, G and sigma are reinitialized.")
    .def("has_gamma", &bob::machine::PLDABase::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("compute_gamma", &bob::machine::PLDABase::computeGamma, (arg("self"), arg("a"), arg("gamma")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_add_gamma", make_function(&bob::machine::PLDABase::getAddGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_gamma", make_function(&bob::machine::PLDABase::getGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Returns the gamma matrix for the given number of samples if it has already been put in cache. Throws an exception otherwise. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("has_log_like_const_term", &bob::machine::PLDABase::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("compute_log_like_const_term", (double (bob::machine::PLDABase::*)(const size_t, const blitz::Array<double,2>&) const)&bob::machine::PLDABase::computeLogLikeConstTerm, (arg("self"), arg("a"), arg("gamma")), "Computes the log likelihood constant term for the given number of samples.")
    .def("get_add_log_like_const_term", &bob::machine::PLDABase::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .def("get_log_like_const_term", &bob::machine::PLDABase::getLogLikeConstTerm, (arg("self"), arg("a")), "Returns the log likelihood constant term for the given number of samples if it has already been put in cache. Throws an exception otherwise.")
    .def("clear_maps", &bob::machine::PLDABase::clearMaps, (arg("self")), "Clear the maps containing the gamma's as well as the log likelihood constant term for few number of samples. These maps are used to make likelihood computations faster.")
    .def("compute_log_likelihood_point_estimate", &py_log_likelihood_point_estimate, (arg("self"), arg("xij"), arg("hi"), arg("wij")), "Computes the log-likelihood of a sample given the latent variables hi and wij (point estimate rather than Bayesian-like full integration).")
    .def(self_ns::str(self_ns::self))
    .add_property("__isigma__", make_function(&bob::machine::PLDABase::getISigma, return_value_policy<copy_const_reference>()), "sigma^{-1} matrix stored in cache")
    .add_property("__alpha__", make_function(&bob::machine::PLDABase::getAlpha, return_value_policy<copy_const_reference>()), "alpha matrix stored in cache")
    .add_property("__beta__", make_function(&bob::machine::PLDABase::getBeta, return_value_policy<copy_const_reference>()), "beta matrix stored in cache")
    .add_property("__ft_beta__", make_function(&bob::machine::PLDABase::getFtBeta, return_value_policy<copy_const_reference>()), "F^T.beta matrix stored in cache")
    .add_property("__gt_i_sigma__", make_function(&bob::machine::PLDABase::getGtISigma, return_value_policy<copy_const_reference>()), "G^T.sigma^{-1} matrix stored in cache")
    .add_property("__logdet_alpha__", &bob::machine::PLDABase::getLogDetAlpha, "Logarithm of the determinant of the alpha matrix stored in cache.")
    .add_property("__logdet_sigma__", &bob::machine::PLDABase::getLogDetSigma, "Logarithm of the determinant of the sigma matrix stored in cache.")
    .def("__precompute__", &bob::machine::PLDABase::precompute, (arg("self")), "Precomputes useful values such as alpha and beta.")
    .def("__precompute_log_like__", &bob::machine::PLDABase::precomputeLogLike, (arg("self")), "Precomputes useful values for log-likelihood computations.")
  ;

  class_<bob::machine::PLDAMachine, boost::shared_ptr<bob::machine::PLDAMachine> >("PLDAMachine", "A PLDAMachine contains class-specific information (from the enrolment samples) when performing Probabilistic Linear Discriminant Analysis (PLDA). It should be attached to a PLDABase that contains information such as the subspaces F and G.\n\nReferences:\n1. 'A Scalable Formulation of Probabilistic Linear Discriminant Analysis: Applied to Face Recognition', Laurent El Shafey, Chris McCool, Roy Wallace, Sebastien Marcel, TPAMI'2013\n2. 'Probabilistic Linear Discriminant Analysis for Inference About Identity', Prince and Elder, ICCV'2007.\n3. 'Probabilistic Models for Inference about Identity', Li, Fu, Mohammed, Elder and Prince, TPAMI'2012.", init<boost::shared_ptr<bob::machine::PLDABase> >((arg("self"), arg("plda_base")), "Builds a new PLDAMachine. An attached PLDABase should be provided, that can be shared by several PLDAMachine."))
    .def(init<>((arg("self")), "Constructs a new empty PLDAMachine."))
    .def(init<bob::io::HDF5File&, boost::shared_ptr<bob::machine::PLDABase> >((arg("self"), arg("config"), arg("plda_base")), "Constructs a new PLDAMachine from a configuration file (and a PLDABase object)."))
    .def(init<const bob::machine::PLDAMachine&>((arg("self"), arg("machine")), "Copy constructs a PLDAMachine"))
    .def(self == self)
    .def(self != self)
    .def("load", &bob::machine::PLDAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::PLDAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("plda_base", &bob::machine::PLDAMachine::getPLDABase, &bob::machine::PLDAMachine::setPLDABase)
    .add_property("dim_d", &bob::machine::PLDAMachine::getDimD, "Dimensionality of the input feature vectors")
    .add_property("dim_f", &bob::machine::PLDAMachine::getDimF, "Dimensionality of the F subspace/matrix of the PLDA model")
    .add_property("dim_g", &bob::machine::PLDAMachine::getDimG, "Dimensionality of the G subspace/matrix of the PLDA model")
    .add_property("n_samples", &bob::machine::PLDAMachine::getNSamples, &bob::machine::PLDAMachine::setNSamples, "Number of enrolled samples")
    .add_property("w_sum_xit_beta_xi", &bob::machine::PLDAMachine::getWSumXitBetaXi, &bob::machine::PLDAMachine::setWSumXitBetaXi)
    .add_property("weighted_sum", make_function(&bob::machine::PLDAMachine::getWeightedSum, return_value_policy<copy_const_reference>()), &bob::machine::PLDAMachine::setWeightedSum)
    .add_property("log_likelihood", &bob::machine::PLDAMachine::getLogLikelihood, &bob::machine::PLDAMachine::setLogLikelihood)
    .def("has_gamma", &bob::machine::PLDAMachine::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_add_gamma", make_function(&bob::machine::PLDAMachine::getAddGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_gamma", make_function(&bob::machine::PLDAMachine::getGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Returns the gamma matrix for the given number of samples if it has already been put in cache. Throws an exception otherwise. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("has_log_like_const_term", &bob::machine::PLDAMachine::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("get_add_log_like_const_term", &bob::machine::PLDAMachine::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .def("get_log_like_const_term", &bob::machine::PLDAMachine::getLogLikeConstTerm, (arg("self"), arg("a")), "Returns the log likelihood constant term for the given number of samples if it has already been put in cache. Throws an exception otherwise.")
    .def("clear_maps", &bob::machine::PLDAMachine::clearMaps, (arg("self")), "Clears the maps containing the gamma's as well as the log likelihood constant term for few number of samples. These maps are used to make likelihood computations faster.")
    .def("compute_log_likelihood", &computeLogLikelihood, computeLogLikelihood_overloads((arg("self"), arg("sample"), arg("use_enrolled_samples")=true), "Computes the log-likelihood considering only the probe sample(s) or jointly the probe sample(s) and the enrolled samples."))
    .def("__call__", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a log-likelihood ratio score.")
    .def("forward", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a log-likelihood ratio score.")
  ;
}
