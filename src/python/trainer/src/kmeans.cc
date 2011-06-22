#include <boost/python.hpp>
#include "io/Arrayset.h"
#include "trainer/KMeansTrainer.h"

using namespace boost::python;
namespace train = Torch::trainer;
namespace mach = Torch::machine;
namespace io = Torch::io;


void bind_trainer_kmeans() {

  typedef train::EMTrainer<mach::KMeansMachine, io::Arrayset> EMTrainerKMeansBase; 

  class_<EMTrainerKMeansBase, boost::noncopyable>("EMTrainerKMeans", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergenceThreshold", &EMTrainerKMeansBase::getConvergenceThreshold, &EMTrainerKMeansBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &EMTrainerKMeansBase::getMaxIterations, &EMTrainerKMeansBase::setMaxIterations, "Max iterations")
    .def("train", &EMTrainerKMeansBase::train, (arg("machine"), arg("data")), "Train a machine using data")
    .def("initialization", &EMTrainerKMeansBase::initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", &EMTrainerKMeansBase::eStep, (arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("mStep", &EMTrainerKMeansBase::mStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
  ;

  class_<train::KMeansTrainer, boost::noncopyable, bases<EMTrainerKMeansBase> >("KMeansTrainer",
      "Trains a KMeans machine.\n"
      "This class implements the expectation-maximisation algorithm for a k-means machine.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006\n"
      "It uses a random initialisation of the means followed by the expectation-maximization algorithm"
      )
    .def(init<optional<double,int> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10)))
    .add_property("seed", &train::KMeansTrainer::getSeed, &train::KMeansTrainer::setSeed, "Seed used to genrated pseudo-random numbers")
  ;

}
