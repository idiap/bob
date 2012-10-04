/**
 * @file trainer/python/overload/gmm_wrappers.cc
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
#include "bob/trainer/GMMTrainer.h"
#include "bob/trainer/MAP_GMMTrainer.h"
#include "bob/trainer/ML_GMMTrainer.h"
#include <boost/shared_ptr.hpp>
#include <limits>

using namespace boost::python;

void deletor(bob::machine::GMMMachine*)
{
}

class EMTrainerGMMWrapper: public bob::trainer::EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> >, 
                           public wrapper<bob::trainer::EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> > > 
{
public:
  EMTrainerGMMWrapper(double convergence_threshold = 0.001, int max_iterations = 10, bool compute_likelihood = true):
    bob::trainer::EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> >(convergence_threshold, max_iterations, compute_likelihood) {}

  virtual ~EMTrainerGMMWrapper() {}
 
  void initialization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("initialization")(machine_ptr, data);
  }
  
  void eStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("e_step")(machine_ptr, data);
  }

  double computeLikelihood(bob::machine::GMMMachine& machine) {
    boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    return this->get_override("compute_likelihood")(machine_ptr);
  }
  
  void mStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("m_step")(machine_ptr, data);
  }

  void finalization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("finalization")(machine_ptr, data);
  }
 
  void train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_train = this->get_override("train")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_train(machine_ptr, data);
    }
    else
      bob::trainer::EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> >::train(machine, data);
  }

  void d_train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> >::train(machine, data);
  }

};


class GMMTrainerWrapper: public bob::trainer::GMMTrainer,
                         public wrapper<bob::trainer::GMMTrainer>
{
public:
  GMMTrainerWrapper(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon()):
    bob::trainer::GMMTrainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {}

  virtual ~GMMTrainerWrapper() {}
  
  void initialization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_initialization = this->get_override("initialization")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_initialization(machine_ptr, data);
    }
    else
      bob::trainer::GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::GMMTrainer::initialization(machine, data);
  }
  
  void eStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_eStep = this->get_override("e_step")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_eStep(machine_ptr, data);
    }
    else
      bob::trainer::GMMTrainer::eStep(machine, data);
  }
  
  void d_eStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::GMMTrainer::eStep(machine, data);
  }

  double computeLikelihood(bob::machine::GMMMachine& machine) {
    if (override python_computeLikelihood = this->get_override("compute_likelihood")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      return python_computeLikelihood(machine_ptr);
    }
    return bob::trainer::GMMTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(bob::machine::GMMMachine& machine) {
    return bob::trainer::GMMTrainer::computeLikelihood(machine);
  }

  void mStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("m_step")(machine_ptr, data);
  }

  void finalization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_finalization = this->get_override("finalization")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_finalization(machine_ptr, data);
    }
    else
      bob::trainer::GMMTrainer::finalization(machine, data);
  }
  
  void d_finalization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::GMMTrainer::finalization(machine, data);
  } 

  void train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_train = this->get_override("train")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_train(machine_ptr, data);
    }
    else
      bob::trainer::GMMTrainer::train(machine, data);
  }

  void d_train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::GMMTrainer::train(machine, data);
  }
 
};


class MAP_GMMTrainerWrapper: public bob::trainer::MAP_GMMTrainer,
                             public wrapper<bob::trainer::MAP_GMMTrainer>
{
public:
  MAP_GMMTrainerWrapper(double relevance_factor = 0, bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon()):
    bob::trainer::MAP_GMMTrainer(relevance_factor, update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {}

  virtual ~MAP_GMMTrainerWrapper() {}

  void initialization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_initialization = this->get_override("initialization")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_initialization(machine_ptr, data);
    }
    else
      bob::trainer::MAP_GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::MAP_GMMTrainer::initialization(machine, data);
  }
  
  void eStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_eStep = this->get_override("e_step")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_eStep(machine_ptr, data);
    }
    else
      bob::trainer::MAP_GMMTrainer::eStep(machine, data);
  }
  
  void d_eStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::MAP_GMMTrainer::eStep(machine, data);
  }

  double computeLikelihood(bob::machine::GMMMachine& machine) {
    if (override python_computeLikelihood = this->get_override("compute_likelihood")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      return python_computeLikelihood(machine_ptr);
    }
    return bob::trainer::MAP_GMMTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(bob::machine::GMMMachine& machine) {
    return bob::trainer::MAP_GMMTrainer::computeLikelihood(machine);
  }

  void mStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_mStep = this->get_override("m_step")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_mStep(machine_ptr, data);
    }
    else
      bob::trainer::MAP_GMMTrainer::mStep(machine, data);
  }
  
  void d_mStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::MAP_GMMTrainer::mStep(machine, data);
  }

  void finalization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_finalization = this->get_override("finalization")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_finalization(machine_ptr, data);
    }
    else
      bob::trainer::MAP_GMMTrainer::finalization(machine, data);
  }
  
  void d_finalization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::MAP_GMMTrainer::finalization(machine, data);
  } 

  void train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_train = this->get_override("train")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_train(machine_ptr, data);
    }
    else
      bob::trainer::MAP_GMMTrainer::train(machine, data);
  }

  void d_train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::MAP_GMMTrainer::train(machine, data);
  }
 
};

class ML_GMMTrainerWrapper: public bob::trainer::ML_GMMTrainer,
                            public wrapper<bob::trainer::ML_GMMTrainer>
{
public:
  ML_GMMTrainerWrapper(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon()):
    bob::trainer::ML_GMMTrainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {}

  virtual ~ML_GMMTrainerWrapper() {}

  void initialization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_initialization = this->get_override("initialization")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_initialization(machine_ptr, data);
    }
    else
      bob::trainer::ML_GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::ML_GMMTrainer::initialization(machine, data);
  }
  
  void eStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_eStep = this->get_override("e_step")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_eStep(machine_ptr, data);
    }
    else
      bob::trainer::ML_GMMTrainer::eStep(machine, data);
  }
  
  void d_eStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::ML_GMMTrainer::eStep(machine, data);
  }

  double computeLikelihood(bob::machine::GMMMachine& machine) {
    if (override python_computeLikelihood = this->get_override("compute_likelihood")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      return python_computeLikelihood(machine_ptr);
    }
    return bob::trainer::ML_GMMTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(bob::machine::GMMMachine& machine) {
    return bob::trainer::ML_GMMTrainer::computeLikelihood(machine);
  }


  void mStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_mStep = this->get_override("m_step")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_mStep(machine_ptr, data);
    }
    else
      bob::trainer::ML_GMMTrainer::mStep(machine, data);
  }
  
  void d_mStep(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::ML_GMMTrainer::mStep(machine, data);
  }

  void finalization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_finalization = this->get_override("finalization")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_finalization(machine_ptr, data);
    }
    else
      bob::trainer::ML_GMMTrainer::finalization(machine, data);
  }
  
  void d_finalization(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::ML_GMMTrainer::finalization(machine, data);
  }
 
  void train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_train = this->get_override("train")) 
    {
      boost::shared_ptr<bob::machine::GMMMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_train(machine_ptr, data);
    }
    else
      bob::trainer::ML_GMMTrainer::train(machine, data);
  }

  void d_train(bob::machine::GMMMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::ML_GMMTrainer::train(machine, data);
  }
 
};


void bind_trainer_gmm_wrappers() {

  typedef bob::trainer::EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> > EMTrainerGMMBase; 

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
    .add_property("convergence_threshold", &bob::trainer::GMMTrainer::getConvergenceThreshold, &bob::trainer::GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &bob::trainer::GMMTrainer::getMaxIterations, &bob::trainer::GMMTrainer::setMaxIterations, "Max iterations")
    .add_property("gmm_statistics", &bob::trainer::GMMTrainer::getGMMStats, &bob::trainer::GMMTrainer::setGMMStats, "The internal GMM statistics. Useful to parallelize the E-step.")
    .def("train", &bob::trainer::GMMTrainer::train, &GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &bob::trainer::GMMTrainer::initialization, &GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &bob::trainer::GMMTrainer::finalization, &GMMTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", &bob::trainer::GMMTrainer::eStep, &GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &bob::trainer::GMMTrainer::computeLikelihood, &GMMTrainerWrapper::d_computeLikelihood, (arg("machine")), "Returns the likelihood")
    .def("m_step", pure_virtual(&bob::trainer::GMMTrainer::mStep), (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;


  class_<MAP_GMMTrainerWrapper, boost::noncopyable >("MAP_GMMTrainer",
      "This class implements the maximum a posteriori M-step "
      "of the expectation-maximisation algorithm for a GMM Machine. "
      "The prior parameters are encoded in the form of a GMM (e.g. a universal background model). "
      "The EM algorithm thus performs GMM adaptation.\n"
      "See Section 3.4 of Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000. We use a \"single adaptation coefficient\", alpha_i, and thus a single relevance factor, r.",
      init<optional<double, bool, bool, bool, double> >((arg("relevance_factor"), arg("update_means"), arg("update_variances"), arg("update_weights"), arg("mean_var_update_responsibilities_threshold"))))
    .def("set_prior_gmm", &bob::trainer::MAP_GMMTrainer::setPriorGMM, 
      "Set the GMM to use as a prior for MAP adaptation. "
      "Generally, this is a \"universal background model\" (UBM), "
      "also referred to as a \"world model\".")
    .add_property("convergence_threshold", &bob::trainer::MAP_GMMTrainer::getConvergenceThreshold, &bob::trainer::MAP_GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &bob::trainer::MAP_GMMTrainer::getMaxIterations, &bob::trainer::MAP_GMMTrainer::setMaxIterations, "Max iterations")
    .add_property("gmm_statistics", &bob::trainer::MAP_GMMTrainer::getGMMStats, &bob::trainer::MAP_GMMTrainer::setGMMStats, "The internal GMM statistics. Useful to parallelize the E-step.")
    .def("train", &bob::trainer::MAP_GMMTrainer::train, &MAP_GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &bob::trainer::MAP_GMMTrainer::initialization, &MAP_GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &bob::trainer::MAP_GMMTrainer::finalization, &MAP_GMMTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", &bob::trainer::MAP_GMMTrainer::eStep, &MAP_GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &bob::trainer::MAP_GMMTrainer::computeLikelihood, &MAP_GMMTrainerWrapper::d_computeLikelihood, (arg("machine")), "Returns the likelihood")
    .def("m_step", &bob::trainer::MAP_GMMTrainer::mStep, &MAP_GMMTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;

   
  class_<ML_GMMTrainerWrapper, boost::noncopyable >("ML_GMMTrainer",
      "This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<optional<bool, bool, bool, double> >((arg("update_means"), arg("update_variances"), arg("update_weights"), arg("mean_var_update_responsibilities_threshold"))))
    .add_property("convergence_threshold", &bob::trainer::ML_GMMTrainer::getConvergenceThreshold, &bob::trainer::ML_GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &bob::trainer::ML_GMMTrainer::getMaxIterations, &bob::trainer::ML_GMMTrainer::setMaxIterations, "Max iterations")
    .add_property("gmm_statistics", &bob::trainer::ML_GMMTrainer::getGMMStats, &bob::trainer::ML_GMMTrainer::setGMMStats, "The internal GMM statistics. Useful to parallelize the E-step.")
    .def("train", &bob::trainer::ML_GMMTrainer::train, &ML_GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &bob::trainer::ML_GMMTrainer::initialization, &ML_GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &bob::trainer::ML_GMMTrainer::finalization, &ML_GMMTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
    .def("e_step", &bob::trainer::ML_GMMTrainer::eStep, &ML_GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &bob::trainer::ML_GMMTrainer::eStep, &ML_GMMTrainerWrapper::d_eStep, (arg("machine")), "Returns the likelihood")
    .def("m_step", &bob::trainer::ML_GMMTrainer::mStep, &ML_GMMTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;

}
