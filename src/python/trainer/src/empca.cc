#include <boost/python.hpp>
#include "io/Arrayset.h"
#include "machine/LinearMachine.h"
#include "trainer/EMPCATrainer.h"

using namespace boost::python;
namespace train = bob::trainer;
namespace mach = bob::machine;
namespace io = bob::io;

object ppca_train(train::EMPCATrainer& t, const io::Arrayset& data) {
  mach::LinearMachine m;
  t.train(m, data);
  return object(m);
}

void bind_trainer_empca() {

  typedef train::EMTrainerNew<mach::LinearMachine, io::Arrayset> EMTrainerLinearBase; 

  class_<EMTrainerLinearBase, boost::noncopyable>("EMTrainerLinear", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergenceThreshold", &EMTrainerLinearBase::getConvergenceThreshold, &EMTrainerLinearBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &EMTrainerLinearBase::getMaxIterations, &EMTrainerLinearBase::setMaxIterations, "Max iterations")
    .add_property("computeLikelihoodVariable", &EMTrainerLinearBase::getComputeLikelihood, &EMTrainerLinearBase::setComputeLikelihood, "Indicates whether the log likelihood should be computed during EM or not")
    .def("train", &EMTrainerLinearBase::train, (arg("machine"), arg("data")), "Trains a machine using data")
    .def("initialization", &EMTrainerLinearBase::initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &EMTrainerLinearBase::finalization, (arg("machine"), arg("data")), "This method is called at the end of the EM algorithm")
    .def("eStep", &EMTrainerLinearBase::eStep, (arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("mStep", &EMTrainerLinearBase::mStep, (arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("computeLikelihood", &EMTrainerLinearBase::computeLikelihood, (arg("machine"), arg("data")), "Computes the current log likelihood given the hidden variable distribution (or the sufficient statistics)")
  ;

  class_<train::EMPCATrainer, boost::noncopyable, bases<EMTrainerLinearBase> >("EMPCATrainer",
      "This class implements the EM algorithm for a Linear Machine (Probabilistic PCA).\n"
      "See Section 12.2 of Bishop, \"Pattern recognition and machine learning\", 2006", init<int,optional<double,double,bool> >((arg("dimensionality"), arg("convergence_threshold"), arg("max_iterations"), arg("compute_likelihood"))))
    .def("train", &ppca_train, (arg("self"), arg("data")), "Trains and returns a Linear machine using the provided data")
    .add_property("seed", &train::EMPCATrainer::getSeed, &train::EMPCATrainer::setSeed, "The seed for the random initialization of W and sigma2")
    .add_property("sigma2", &train::EMPCATrainer::getSigma2, &train::EMPCATrainer::setSigma2, "The noise sigma2 of the probabilistic model")
  ;

}
