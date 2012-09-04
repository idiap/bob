/**
 * @file python/trainer/src/gmm_wrappers.cc
 * @date Thu Jun 9 18:12:33 2011 +0200
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
#include "io/Arrayset.h"
#include "trainer/GMMTrainer.h"
#include "trainer/MAP_GMMTrainer.h"
#include "trainer/ML_GMMTrainer.h"
#include <limits>

using namespace boost::python;
namespace train = bob::trainer;
namespace mach = bob::machine;
namespace io = bob::io;


class EMTrainerGMMWrapper: public train::EMTrainer<mach::GMMMachine, io::Arrayset>, 
                           public wrapper<train::EMTrainer<mach::GMMMachine, io::Arrayset> > 
{
public:
  EMTrainerGMMWrapper(double convergence_threshold = 0.001, int max_iterations = 10, bool compute_likelihood = true):
    train::EMTrainer<mach::GMMMachine, io::Arrayset >(convergence_threshold, max_iterations, compute_likelihood) {}

  virtual ~EMTrainerGMMWrapper() {}
 
  void initialization(mach::GMMMachine& machine, const io::Arrayset& data) {
    this->get_override("initialization")(machine, data);
  }
  
  void eStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    this->get_override("e_step")(machine, data);
  }

  double computeLikelihood(mach::GMMMachine& machine) {
    return this->get_override("compute_likelihood")(machine);
  }
  
  void mStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    this->get_override("m_step")(machine, data);
  }

  void finalization(mach::GMMMachine& machine, const io::Arrayset& data) {
    this->get_override("finalization")(machine, data);
  }
 
  void train(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::EMTrainer<mach::GMMMachine, io::Arrayset>::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::EMTrainer<mach::GMMMachine, io::Arrayset>::train(machine, data);
  }

};


class GMMTrainerWrapper: public train::GMMTrainer,
                         public wrapper<train::GMMTrainer>
{
public:
  GMMTrainerWrapper(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon()):
    train::GMMTrainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {}

  virtual ~GMMTrainerWrapper() {}
  
  void initialization(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_initialization = this->get_override("initialization")) 
      python_initialization(machine, data);
    else
      train::GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::GMMTrainer::initialization(machine, data);
  }
  
  void eStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_eStep = this->get_override("e_step")) python_eStep(machine, data);
    train::GMMTrainer::eStep(machine, data);
  }
  
  void d_eStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::GMMTrainer::eStep(machine, data);
  }

  double computeLikelihood(mach::GMMMachine& machine) {
    if (override python_computeLikelihood = this->get_override("compute_likelihood")) return python_computeLikelihood(machine);
    return train::GMMTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(mach::GMMMachine& machine) {
    return train::GMMTrainer::computeLikelihood(machine);
  }

  void mStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    this->get_override("m_step")(machine, data);
  }

  void finalization(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_finalization = this->get_override("finalization")) 
      python_finalization(machine, data);
    else
      train::GMMTrainer::finalization(machine, data);
  }
  
  void d_finalization(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::GMMTrainer::finalization(machine, data);
  } 

  void train(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::GMMTrainer::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::GMMTrainer::train(machine, data);
  }
 
};


class MAP_GMMTrainerWrapper: public train::MAP_GMMTrainer,
                             public wrapper<train::MAP_GMMTrainer>
{
public:
  MAP_GMMTrainerWrapper(double relevance_factor = 0, bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon()):
    train::MAP_GMMTrainer(relevance_factor, update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {}

  virtual ~MAP_GMMTrainerWrapper() {}

  void initialization(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_initialization = this->get_override("initialization")) 
      python_initialization(machine, data);
    else
      train::MAP_GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::MAP_GMMTrainer::initialization(machine, data);
  }
  
  void eStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_eStep = this->get_override("e_step")) python_eStep(machine, data);
    train::MAP_GMMTrainer::eStep(machine, data);
  }
  
  void d_eStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::MAP_GMMTrainer::eStep(machine, data);
  }

  double computeLikelihood(mach::GMMMachine& machine) {
    if (override python_computeLikelihood = this->get_override("compute_likelihood")) return python_computeLikelihood(machine);
    return train::MAP_GMMTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(mach::GMMMachine& machine) {
    return train::MAP_GMMTrainer::computeLikelihood(machine);
  }

  void mStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_mStep = this->get_override("m_step")) 
      python_mStep(machine, data);
    else
      train::MAP_GMMTrainer::mStep(machine, data);
  }
  
  void d_mStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::MAP_GMMTrainer::mStep(machine, data);
  }

  void finalization(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_finalization = this->get_override("finalization")) 
      python_finalization(machine, data);
    else
      train::MAP_GMMTrainer::finalization(machine, data);
  }
  
  void d_finalization(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::MAP_GMMTrainer::finalization(machine, data);
  } 

  void train(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::MAP_GMMTrainer::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::MAP_GMMTrainer::train(machine, data);
  }
 
};

class ML_GMMTrainerWrapper: public train::ML_GMMTrainer,
                            public wrapper<train::ML_GMMTrainer>
{
public:
  ML_GMMTrainerWrapper(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon()):
    train::ML_GMMTrainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {}

  virtual ~ML_GMMTrainerWrapper() {}

  void initialization(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_initialization = this->get_override("initialization")) 
      python_initialization(machine, data);
    else
      train::ML_GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::ML_GMMTrainer::initialization(machine, data);
  }
  
  void eStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_eStep = this->get_override("e_step")) python_eStep(machine, data);
    train::ML_GMMTrainer::eStep(machine, data);
  }
  
  void d_eStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::ML_GMMTrainer::eStep(machine, data);
  }

  double computeLikelihood(mach::GMMMachine& machine) {
    if (override python_computeLikelihood = this->get_override("compute_likelihood")) return python_computeLikelihood(machine);
    return train::ML_GMMTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(mach::GMMMachine& machine) {
    return train::ML_GMMTrainer::computeLikelihood(machine);
  }


  void mStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_mStep = this->get_override("m_step")) 
      python_mStep(machine, data);
    else
      train::ML_GMMTrainer::mStep(machine, data);
  }
  
  void d_mStep(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::ML_GMMTrainer::mStep(machine, data);
  }

  void finalization(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_finalization = this->get_override("finalization")) 
      python_finalization(machine, data);
    else
      train::ML_GMMTrainer::finalization(machine, data);
  }
  
  void d_finalization(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::ML_GMMTrainer::finalization(machine, data);
  }
 
  void train(mach::GMMMachine& machine, const io::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::ML_GMMTrainer::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const io::Arrayset& data) {
    train::ML_GMMTrainer::train(machine, data);
  }
 
};


void bind_trainer_gmm_wrappers() {

  typedef train::EMTrainer<mach::GMMMachine, io::Arrayset> EMTrainerGMMBase; 

  class_<EMTrainerGMMWrapper, boost::noncopyable >("EMTrainerGMM", no_init)
    .def(init<optional<double, int, bool> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=true)))
    .add_property("convergence_threshold", &EMTrainerGMMBase::getConvergenceThreshold, &EMTrainerGMMBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerGMMBase::getMaxIterations, &EMTrainerGMMBase::setMaxIterations, "Max iterations")
    .def("train", &EMTrainerGMMBase::train, &EMTrainerGMMWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using data")
    .def("initialization", pure_virtual(&EMTrainerGMMBase::initialization), (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", pure_virtual(&EMTrainerGMMBase::finalization), (arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", pure_virtual(&EMTrainerGMMBase::eStep), (arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("compute_likelihood", pure_virtual(&EMTrainerGMMBase::computeLikelihood), (arg("machine")), "Returns the likelihood")
    .def("m_step", pure_virtual(&EMTrainerGMMBase::mStep), (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
  ;


  class_<GMMTrainerWrapper, boost::noncopyable >("GMMTrainer",
      "This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<optional<bool, bool, bool, double> >((arg("update_means"), arg("update_variances"), arg("update_weights"), arg("mean_var_update_responsibilities_threshold"))))
    .add_property("convergence_threshold", &train::GMMTrainer::getConvergenceThreshold, &train::GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &train::GMMTrainer::getMaxIterations, &train::GMMTrainer::setMaxIterations, "Max iterations")
    .def("train", &train::GMMTrainer::train, &GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &train::GMMTrainer::initialization, &GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &train::GMMTrainer::finalization, &GMMTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", &train::GMMTrainer::eStep, &GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &train::GMMTrainer::computeLikelihood, &GMMTrainerWrapper::d_computeLikelihood, (arg("machine")), "Returns the likelihood")
    .def("m_step", pure_virtual(&train::GMMTrainer::mStep), (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;


  class_<MAP_GMMTrainerWrapper, boost::noncopyable >("MAP_GMMTrainer",
      "This class implements the maximum a posteriori M-step "
      "of the expectation-maximisation algorithm for a GMM Machine. "
      "The prior parameters are encoded in the form of a GMM (e.g. a universal background model). "
      "The EM algorithm thus performs GMM adaptation.\n"
      "See Section 3.4 of Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000. We use a \"single adaptation coefficient\", alpha_i, and thus a single relevance factor, r.",
      init<optional<double, bool, bool, bool, double> >((arg("relevance_factor"), arg("update_means"), arg("update_variances"), arg("update_weights"), arg("mean_var_update_responsibilities_threshold"))))
    .def("set_prior_gmm", &train::MAP_GMMTrainer::setPriorGMM, 
      "Set the GMM to use as a prior for MAP adaptation. "
      "Generally, this is a \"universal background model\" (UBM), "
      "also referred to as a \"world model\".")
    .add_property("convergence_threshold", &train::MAP_GMMTrainer::getConvergenceThreshold, &train::MAP_GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &train::MAP_GMMTrainer::getMaxIterations, &train::MAP_GMMTrainer::setMaxIterations, "Max iterations")
    .def("train", &train::MAP_GMMTrainer::train, &MAP_GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &train::MAP_GMMTrainer::initialization, &MAP_GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &train::MAP_GMMTrainer::finalization, &MAP_GMMTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", &train::MAP_GMMTrainer::eStep, &MAP_GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &train::MAP_GMMTrainer::computeLikelihood, &MAP_GMMTrainerWrapper::d_computeLikelihood, (arg("machine")), "Returns the likelihood")
    .def("m_step", &train::MAP_GMMTrainer::mStep, &MAP_GMMTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;

   
  class_<ML_GMMTrainerWrapper, boost::noncopyable >("ML_GMMTrainer",
      "This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<optional<bool, bool, bool, double> >((arg("update_means"), arg("update_variances"), arg("update_weights"), arg("mean_var_update_responsibilities_threshold"))))
    .add_property("convergence_threshold", &train::ML_GMMTrainer::getConvergenceThreshold, &train::ML_GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &train::ML_GMMTrainer::getMaxIterations, &train::ML_GMMTrainer::setMaxIterations, "Max iterations")
    .def("train", &train::ML_GMMTrainer::train, &ML_GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &train::ML_GMMTrainer::initialization, &ML_GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &train::ML_GMMTrainer::finalization, &ML_GMMTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", &train::ML_GMMTrainer::eStep, &ML_GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &train::ML_GMMTrainer::eStep, &ML_GMMTrainerWrapper::d_eStep, (arg("machine")), "Returns the likelihood")
    .def("m_step", &train::ML_GMMTrainer::mStep, &ML_GMMTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;

}
