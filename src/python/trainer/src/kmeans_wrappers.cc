/**
 * @file src/python/trainer/src/kmeans_wrappers.cc
 * @date Thu Jun 9 18:12:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "trainer/KMeansTrainer.h"

using namespace boost::python;
namespace train = bob::trainer;
namespace mach = bob::machine;
namespace io = bob::io;


class EMTrainerKMeansWrapper: public train::EMTrainerNew<mach::KMeansMachine, io::Arrayset>, 
                              public wrapper<train::EMTrainerNew<mach::KMeansMachine, io::Arrayset> > 
{
public:
  EMTrainerKMeansWrapper(double convergence_threshold = 0.001, int max_iterations = 10, bool compute_likelihood=true):
    train::EMTrainerNew<mach::KMeansMachine, io::Arrayset >(convergence_threshold, max_iterations, compute_likelihood) {}

  virtual ~EMTrainerKMeansWrapper() {}
 
  virtual void initialization(mach::KMeansMachine& machine, const io::Arrayset& data) {
    this->get_override("initialization")(machine, data);
  }
  
  virtual void eStep(mach::KMeansMachine& machine, const io::Arrayset& data) {
    this->get_override("eStep")(machine, data);
  }
  
  virtual void mStep(mach::KMeansMachine& machine, const io::Arrayset& data) {
    this->get_override("mStep")(machine, data);
  }

  virtual double computeLikelihood(mach::KMeansMachine& machine) {
    return this->get_override("computeLikelihood")(machine);
  }

  virtual void finalization(mach::KMeansMachine& machine, const io::Arrayset& data) {
    this->get_override("finalization")(machine, data);
  }
 
  virtual void train(mach::KMeansMachine& machine, const io::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::EMTrainerNew<mach::KMeansMachine, io::Arrayset>::train(machine, data);
  }

  virtual void d_train(mach::KMeansMachine& machine, const io::Arrayset& data) {
    train::EMTrainerNew<mach::KMeansMachine, io::Arrayset>::train(machine, data);
  }

};

class KMeansTrainerWrapper: public train::KMeansTrainer,
                            public wrapper<train::KMeansTrainer>
{
public:
  KMeansTrainerWrapper(double convergence_threshold = 0.001, int max_iterations = 10, bool compute_likelihood = true):
    train::KMeansTrainer(convergence_threshold, max_iterations, compute_likelihood) {}

  virtual ~KMeansTrainerWrapper() {}
 
  void initialization(mach::KMeansMachine& machine, const io::Arrayset& data) {
    if (override python_initialization = this->get_override("initialization"))
      python_initialization(machine, data);
    else
      train::KMeansTrainer::initialization(machine, data);
  }
  
  void d_initialization(mach::KMeansMachine& machine, const io::Arrayset& data) {
    train::KMeansTrainer::initialization(machine, data);
  }
  
  void eStep(mach::KMeansMachine& machine, const io::Arrayset& data) {
    if (override python_eStep = this->get_override("eStep")) python_eStep(machine, data);
    train::KMeansTrainer::eStep(machine, data);
  }
  
  void d_eStep(mach::KMeansMachine& machine, const io::Arrayset& data) {
    train::KMeansTrainer::eStep(machine, data);
  }
  
  void mStep(mach::KMeansMachine& machine, const io::Arrayset& data) {
    if (override python_mStep = this->get_override("mStep")) 
      python_mStep(machine, data);
    else
      train::KMeansTrainer::mStep(machine, data);
  }

  void d_mStep(mach::KMeansMachine& machine, const io::Arrayset& data) {
    train::KMeansTrainer::mStep(machine, data);
  }

  double computeLikelihood(mach::KMeansMachine& machine) {
    if (override python_computeLikelihood = this->get_override("computeLikelihood")) return python_computeLikelihood(machine);
    return train::KMeansTrainer::computeLikelihood(machine);
  }
  
  double d_computeLikelihood(mach::KMeansMachine& machine) {
    return train::KMeansTrainer::computeLikelihood(machine);
  }

  void finalization(mach::KMeansMachine& machine, const io::Arrayset& data) {
    if (override python_finalization = this->get_override("finalization"))
      python_finalization(machine, data);
    else
      train::KMeansTrainer::finalization(machine, data);
  }
  
  void d_finalization(mach::KMeansMachine& machine, const io::Arrayset& data) {
    train::KMeansTrainer::finalization(machine, data);
  }
  
  void train(mach::KMeansMachine& machine, const io::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::KMeansTrainer::train(machine, data);
  }

  void d_train(mach::KMeansMachine& machine, const io::Arrayset& data) {
    train::KMeansTrainer::train(machine, data);
  }

};


void bind_trainer_kmeans_wrappers() {

  typedef train::EMTrainerNew<mach::KMeansMachine, io::Arrayset> EMTrainerKMeansBase; 

  class_<EMTrainerKMeansWrapper, boost::noncopyable>("EMTrainerKMeans", no_init)
    .def(init<optional<double, int, bool> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=true)))
    .add_property("convergenceThreshold", &EMTrainerKMeansBase::getConvergenceThreshold, &EMTrainerKMeansBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &EMTrainerKMeansBase::getMaxIterations, &EMTrainerKMeansBase::setMaxIterations, "Max iterations")
    .add_property("computeLikelihood", &EMTrainerKMeansBase::getComputeLikelihood, &EMTrainerKMeansBase::setComputeLikelihood, "Tells whether we compute the likelihood or not")
    .def("train", &EMTrainerKMeansBase::train, &EMTrainerKMeansWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using data")
    .def("initialization", pure_virtual(&EMTrainerKMeansBase::initialization), (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", pure_virtual(&EMTrainerKMeansBase::eStep), (arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("mStep", pure_virtual(&EMTrainerKMeansBase::mStep), (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("computeLikelihood", pure_virtual(&EMTrainerKMeansBase::computeLikelihood), (arg("machine")), "Returns the average min distance.")
    .def("finalization", pure_virtual(&EMTrainerKMeansBase::finalization), (arg("machine"), arg("data")), "This method is called after the EM algorithm")
  ;

  class_<KMeansTrainerWrapper, boost::noncopyable>("KMeansTrainer",
      "Trains a KMeans machine.\n"
      "This class implements the expectation-maximisation algorithm for a k-means machine.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006\n"
      "It uses a random initialisation of the means followed by the expectation-maximization algorithm"
      )
    .def(init<optional<double,int,bool> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=true)))
    .add_property("convergenceThreshold", &KMeansTrainerWrapper::getConvergenceThreshold, &KMeansTrainerWrapper::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &KMeansTrainerWrapper::getMaxIterations, &KMeansTrainerWrapper::setMaxIterations, "Max iterations")
    .add_property("seed", &KMeansTrainerWrapper::getSeed, &KMeansTrainerWrapper::setSeed, "Seed used to genrated pseudo-random numbers")
    .def("train", &train::KMeansTrainer::train, &KMeansTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &train::KMeansTrainer::initialization, &KMeansTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", &train::KMeansTrainer::eStep, &KMeansTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("mStep", &train::KMeansTrainer::mStep, &KMeansTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
    .def("computeLikelihood", &train::KMeansTrainer::computeLikelihood, &KMeansTrainerWrapper::d_computeLikelihood, (arg("machine")), "Returns the average min distance.")
    .def("finalization", &train::KMeansTrainer::finalization, &KMeansTrainerWrapper::d_finalization, (arg("machine"), arg("data")), "This method is called after the EM algorithm")
  ;


  }
