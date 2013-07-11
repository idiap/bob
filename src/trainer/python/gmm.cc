/**
 * @file trainer/python/gmm.cc
 * @date Thu Jun 9 18:12:33 2011 +0200
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
#include <bob/trainer/GMMTrainer.h>
#include <bob/trainer/MAP_GMMTrainer.h>
#include <bob/trainer/ML_GMMTrainer.h>

using namespace boost::python;

typedef bob::trainer::EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> > EMTrainerGMMBase; 

static void py_train(EMTrainerGMMBase& trainer, bob::machine::GMMMachine& machine, bob::python::const_ndarray sample)
{
  trainer.train(machine, sample.bz<double,2>());
}

static void py_initialize(EMTrainerGMMBase& trainer, bob::machine::GMMMachine& machine, bob::python::const_ndarray sample)
{
  trainer.initialize(machine, sample.bz<double,2>());
}

static void py_finalize(EMTrainerGMMBase& trainer, bob::machine::GMMMachine& machine, bob::python::const_ndarray sample)
{
  trainer.finalize(machine, sample.bz<double,2>());
}

static void py_eStep(EMTrainerGMMBase& trainer, bob::machine::GMMMachine& machine, bob::python::const_ndarray sample)
{
  trainer.eStep(machine, sample.bz<double,2>());
}

static void py_mStep(EMTrainerGMMBase& trainer, bob::machine::GMMMachine& machine, bob::python::const_ndarray sample)
{
  trainer.mStep(machine, sample.bz<double,2>());
}

void bind_trainer_gmm() {

  class_<EMTrainerGMMBase, boost::noncopyable>("EMTrainerGMM", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergence_threshold", &EMTrainerGMMBase::getConvergenceThreshold, &EMTrainerGMMBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerGMMBase::getMaxIterations, &EMTrainerGMMBase::setMaxIterations, "Max iterations")
    .def("train", &py_train, (arg("self"), arg("machine"), arg("data")), "Train a machine using data")
    .def("initialize", &py_initialize, (arg("self"), arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalize", &py_finalize, (arg("self"), arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", &py_eStep, (arg("self"), arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("m_step", &py_mStep, (arg("self"), arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &EMTrainerGMMBase::computeLikelihood, (arg("self"), arg("machine")), "Returns the likelihood.")
  ;

  class_<bob::trainer::GMMTrainer, boost::noncopyable, bases<EMTrainerGMMBase> >("GMMTrainer",
      "This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006", no_init)
    .add_property("gmm_statistics", make_function(&bob::trainer::GMMTrainer::getGMMStats, return_value_policy<copy_const_reference>()), &bob::trainer::GMMTrainer::setGMMStats, "The internal GMM statistics. Useful to parallelize the E-step.")
  ;

  class_<bob::trainer::MAP_GMMTrainer, boost::noncopyable, bases<bob::trainer::GMMTrainer> >("MAP_GMMTrainer",
      "This class implements the maximum a posteriori M-step "
      "of the expectation-maximisation algorithm for a GMM Machine. "
      "The prior parameters are encoded in the form of a GMM (e.g. a universal background model). "
      "The EM algorithm thus performs GMM adaptation.\n"
      "See Section 3.4 of Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000. We use a \"single adaptation coefficient\", alpha_i, and thus a single relevance factor, r.",
      init<optional<const double, const bool, const bool, const bool, const double> >((arg("self"), arg("relevance_factor"), arg("update_means"), arg("update_variances"), arg("update_weights"), arg("responsibilities_threshold"))))
    .def("set_prior_gmm", &bob::trainer::MAP_GMMTrainer::setPriorGMM, (arg("self"), arg("prior_gmm")), 
      "Set the GMM to use as a prior for MAP adaptation. "
      "Generally, this is a \"universal background model\" (UBM), "
      "also referred to as a \"world model\".")
    .def("set_t3_map", &bob::trainer::MAP_GMMTrainer::setT3MAP, (arg("self"), arg("alpha")),
      "Use a torch3-like MAP adaptation rule instead of Reynolds'one.")
    .def("unset_t3_map", &bob::trainer::MAP_GMMTrainer::unsetT3MAP, (arg("self")),
      "Use a Reynolds' MAP adaptation (rather than torch3-like).")
  ;
 
  class_<bob::trainer::ML_GMMTrainer, boost::noncopyable, bases<bob::trainer::GMMTrainer> >("ML_GMMTrainer",
      "This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<optional<const bool, const bool, const bool, const double> >((arg("self"), arg("update_means"), arg("update_variances"), arg("update_weights"), arg("responsibilities_threshold"))))
  ;
}
