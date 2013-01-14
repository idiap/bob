/**
 * @file trainer/python/overload/kmeans_wrappers.cc
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
#include "bob/trainer/KMeansTrainer.h"
#include <boost/shared_ptr.hpp>

using namespace boost::python;

void deletor(bob::machine::KMeansMachine*)
{
}

class EMTrainerKMeansWrapper: public bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >, 
                              public wrapper<bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> > > 
{
public:
  EMTrainerKMeansWrapper(double convergence_threshold = 0.001, int max_iterations = 10, bool compute_likelihood=true):
    bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >(convergence_threshold, max_iterations, compute_likelihood) {}

  virtual ~EMTrainerKMeansWrapper() {}
 
  virtual void initialization(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("initialization")(machine_ptr, data);
  }
  
  virtual void eStep(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("e_step")(machine_ptr, data);
  }
  
  virtual void mStep(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("m_step")(machine_ptr, data);
  }

  virtual double computeLikelihood(bob::machine::KMeansMachine& machine) {
    boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    return this->get_override("compute_likelihood")(machine_ptr);
  }

  virtual void finalization(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
    this->get_override("finalization")(machine_ptr, data);
  }
 
  virtual void train(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_train = this->get_override("train"))
    { 
      boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_train(machine_ptr, data);
    }
    else
      bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >::train(machine, data);
  }

  virtual void d_train(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >::train(machine, data);
  }

};

class KMeansTrainerWrapper: public bob::trainer::KMeansTrainer,
                            public wrapper<bob::trainer::KMeansTrainer>
{
public:
  KMeansTrainerWrapper(double convergence_threshold = 0.001, int max_iterations = 10, bool compute_likelihood = true):
    bob::trainer::KMeansTrainer(convergence_threshold, max_iterations, compute_likelihood) {}

  virtual ~KMeansTrainerWrapper() {}
 
  void initialization(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_initialization = this->get_override("initialization"))
    {
      boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_initialization(machine_ptr, data);
    }
    else
      bob::trainer::KMeansTrainer::initialization(machine, data);
  }
  
  void d_initialization(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::KMeansTrainer::initialization(machine, data);
  }
  
  void eStep(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_eStep = this->get_override("e_step")) 
    {
      boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_eStep(machine_ptr, data);
    }
    else
      bob::trainer::KMeansTrainer::eStep(machine, data);
  }
  
  void d_eStep(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::KMeansTrainer::eStep(machine, data);
  }
  
  void mStep(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_mStep = this->get_override("m_step")) 
    {
      boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_mStep(machine_ptr, data);
    }
    else
      bob::trainer::KMeansTrainer::mStep(machine, data);
  }

  void d_mStep(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::KMeansTrainer::mStep(machine, data);
  }

  double computeLikelihood(bob::machine::KMeansMachine& machine) {
    if (override python_computeLikelihood = this->get_override("compute_likelihood")) 
    {
      boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      return python_computeLikelihood(machine_ptr);
    }
    return bob::trainer::KMeansTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(bob::machine::KMeansMachine& machine) {
    return bob::trainer::KMeansTrainer::computeLikelihood(machine);
  }

  void finalization(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_finalization = this->get_override("finalization"))
    {
      boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_finalization(machine_ptr, data);
    }
    else
      bob::trainer::KMeansTrainer::finalization(machine, data);
  }
  
  void d_finalization(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::KMeansTrainer::finalization(machine, data);
  }
  
  void train(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    if (override python_train = this->get_override("train")) 
    {
      boost::shared_ptr<bob::machine::KMeansMachine> machine_ptr(&machine, std::ptr_fun(deletor));
      python_train(machine_ptr, data);
    }
    else
      bob::trainer::KMeansTrainer::train(machine, data);
  }

  void d_train(bob::machine::KMeansMachine& machine, const blitz::Array<double,2>& data) {
    bob::trainer::KMeansTrainer::train(machine, data);
  }

};


void bind_trainer_kmeans_wrappers() {

  typedef bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> > EMTrainerKMeansBase; 

  class_<EMTrainerKMeansWrapper, boost::noncopyable>("EMTrainerKMeans", no_init)
    .def(init<optional<double, int, bool> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=true)))
    .add_property("convergence_threshold", &EMTrainerKMeansBase::getConvergenceThreshold, &EMTrainerKMeansBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerKMeansBase::getMaxIterations, &EMTrainerKMeansBase::setMaxIterations, "Max iterations")
    .add_property("compute_likelihood", &EMTrainerKMeansBase::getComputeLikelihood, &EMTrainerKMeansBase::setComputeLikelihood, "Tells whether we compute the likelihood or not")
    .def("train", &EMTrainerKMeansBase::train, &EMTrainerKMeansWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using data")
    .def("initialization", pure_virtual(&EMTrainerKMeansBase::initialization), (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("e_step", pure_virtual(&EMTrainerKMeansBase::eStep), (arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("m_step", pure_virtual(&EMTrainerKMeansBase::mStep), (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", pure_virtual(&EMTrainerKMeansBase::computeLikelihood), (arg("machine")), "Returns the average min distance.")
    .def("finalization", pure_virtual(&EMTrainerKMeansBase::finalization), (arg("machine"), arg("data")), "This method is called after the EM algorithm")
  ;

  class_<KMeansTrainerWrapper, boost::noncopyable>("KMeansTrainer",
      "Trains a KMeans machine.\n"
      "This class implements the expectation-maximisation algorithm for a k-means machine.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006\n"
      "It uses a random initialisation of the means followed by the expectation-maximization algorithm"
      )
    .def(init<optional<double,int,bool> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=true)))
    .add_property("convergence_threshold", &KMeansTrainerWrapper::getConvergenceThreshold, &KMeansTrainerWrapper::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &KMeansTrainerWrapper::getMaxIterations, &KMeansTrainerWrapper::setMaxIterations, "Max iterations")
    .add_property("seed", &KMeansTrainerWrapper::getSeed, &KMeansTrainerWrapper::setSeed, "Seed used to genrated pseudo-random numbers")
    .def("train", &bob::trainer::KMeansTrainer::train, &KMeansTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &bob::trainer::KMeansTrainer::initialization, &KMeansTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("e_step", &bob::trainer::KMeansTrainer::eStep, &KMeansTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("m_step", &bob::trainer::KMeansTrainer::mStep, &KMeansTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
    .def("compute_likelihood", &bob::trainer::KMeansTrainer::computeLikelihood, &KMeansTrainerWrapper::d_computeLikelihood, (arg("machine")), "Returns the average min distance.")
    .def("finalization", &bob::trainer::KMeansTrainer::finalization, &KMeansTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
  ;


  }
