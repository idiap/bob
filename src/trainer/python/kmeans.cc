/**
 * @file trainer/python/kmeans.cc
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

#include <bob/python/ndarray.h>
#include <bob/trainer/KMeansTrainer.h>

using namespace boost::python;

typedef bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> > EMTrainerKMeansBase; 

static void py_setZeroethOrderStats(bob::trainer::KMeansTrainer& op, bob::python::const_ndarray stats) {
  const bob::core::array::typeinfo& info = stats.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  op.setZeroethOrderStats(stats.bz<double,1>());
}

static void py_setFirstOrderStats(bob::trainer::KMeansTrainer& op, bob::python::const_ndarray stats) {
  const bob::core::array::typeinfo& info = stats.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  op.setFirstOrderStats(stats.bz<double,2>());
}

static void py_train(EMTrainerKMeansBase& trainer, 
  bob::machine::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.train(machine, sample.bz<double,2>());
}

static void py_initialize(EMTrainerKMeansBase& trainer, 
  bob::machine::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.initialize(machine, sample.bz<double,2>());
}

static void py_finalize(EMTrainerKMeansBase& trainer, 
  bob::machine::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.finalize(machine, sample.bz<double,2>());
}

static void py_eStep(EMTrainerKMeansBase& trainer, 
  bob::machine::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.eStep(machine, sample.bz<double,2>());
}

static void py_mStep(EMTrainerKMeansBase& trainer, 
  bob::machine::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.mStep(machine, sample.bz<double,2>());
}

void bind_trainer_kmeans() 
{
  class_<EMTrainerKMeansBase, boost::noncopyable>("EMTrainerKMeans", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergence_threshold", &EMTrainerKMeansBase::getConvergenceThreshold, &EMTrainerKMeansBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerKMeansBase::getMaxIterations, &EMTrainerKMeansBase::setMaxIterations, "Max iterations")
    .add_property("compute_likelihood", &EMTrainerKMeansBase::getComputeLikelihood, &EMTrainerKMeansBase::setComputeLikelihood, "Tells whether we compute the average min (square Euclidean) distance or not.")
    .add_property("rng", &EMTrainerKMeansBase::getRng, &EMTrainerKMeansBase::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .def(self == self)
    .def(self != self)
    .def("train", &py_train, (arg("self"), arg("machine"), arg("data")), "Train a machine using data")
    .def("initialize", &py_initialize, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("e_step", &py_eStep, (arg("self"), arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("m_step", &py_mStep, (arg("self"), arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &EMTrainerKMeansBase::computeLikelihood, (arg("self"), arg("machine")), "Returns the average min (square Euclidean) distance")
    .def("finalize", &py_finalize, (arg("self"), arg("machine"), arg("data")), "This method is called after the EM algorithm")
  ;

  // Starts binding the KMeansTrainer
  class_<bob::trainer::KMeansTrainer, boost::shared_ptr<bob::trainer::KMeansTrainer>, boost::noncopyable, bases<EMTrainerKMeansBase> > KMT("KMeansTrainer",
      "Trains a KMeans machine.\n"
      "This class implements the expectation-maximisation algorithm for a k-means machine.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006\n"
      "It uses a random initialisation of the means followed by the expectation-maximization algorithm"
      );

  // Binds methods that does not have nested enum values as default parameters
  KMT.def(self == self)
     .def(self != self)
     .add_property("initialization_method", &bob::trainer::KMeansTrainer::getInitializationMethod, &bob::trainer::KMeansTrainer::setInitializationMethod, "The initialization method to generate the initial means.")
     .add_property("rng", &bob::trainer::KMeansTrainer::getRng, &bob::trainer::KMeansTrainer::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of the means.")
     .add_property("average_min_distance", &bob::trainer::KMeansTrainer::getAverageMinDistance, &bob::trainer::KMeansTrainer::setAverageMinDistance, "Average min (square Euclidean) distance. Useful to parallelize the E-step.")
     .add_property("zeroeth_order_statistics", make_function(&bob::trainer::KMeansTrainer::getZeroethOrderStats, return_value_policy<copy_const_reference>()), &py_setZeroethOrderStats, "The zeroeth order statistics. Useful to parallelize the E-step.")
     .add_property("first_order_statistics", make_function(&bob::trainer::KMeansTrainer::getFirstOrderStats, return_value_policy<copy_const_reference>()), &py_setFirstOrderStats, "The first order statistics. Useful to parallelize the E-step.")
    ;

  // Sets the scope to the one of the KMeansTrainer
  scope s(KMT);

  // Adds enum in the previously defined current scope
  enum_<bob::trainer::KMeansTrainer::InitializationMethod>("initialization_method_type")
    .value("RANDOM", bob::trainer::KMeansTrainer::RANDOM)
    .value("RANDOM_NO_DUPLICATE", bob::trainer::KMeansTrainer::RANDOM_NO_DUPLICATE)
#if BOOST_VERSION >= 104700
    .value("KMEANS_PLUS_PLUS", bob::trainer::KMeansTrainer::KMEANS_PLUS_PLUS)
#endif
    .export_values()
    ;   

  // Binds methods that has nested enum values as default parameters
  KMT.def(init<optional<double,int,bool,bob::trainer::KMeansTrainer::InitializationMethod> >((arg("self"), arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=true, arg("initialization_method")=bob::trainer::KMeansTrainer::RANDOM)));
}
