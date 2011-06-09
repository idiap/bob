#include <boost/python.hpp>
#include "database/Arrayset.h"
#include "trainer/GMMTrainer.h"
#include "trainer/MAP_GMMTrainer.h"
#include "trainer/ML_GMMTrainer.h"

using namespace boost::python;
namespace train = Torch::trainer;
namespace mach = Torch::machine;
namespace db = Torch::database;

void bind_trainer_gmm() {

  typedef train::EMTrainer<mach::GMMMachine, db::Arrayset> EMTrainerGMMBase; 

  class_<EMTrainerGMMBase, boost::noncopyable>("EMTrainerGMM", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergenceThreshold", &EMTrainerGMMBase::getConvergenceThreshold, &EMTrainerGMMBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &EMTrainerGMMBase::getMaxIterations, &EMTrainerGMMBase::setMaxIterations, "Max iterations")
    .def("train", &EMTrainerGMMBase::train, (arg("machine"), arg("data")), "Train a machine using data")
    .def("initialization", &EMTrainerGMMBase::initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", &EMTrainerGMMBase::eStep, (arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("mStep", &EMTrainerGMMBase::mStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
  ;

  class_<train::GMMTrainer, boost::noncopyable, bases<EMTrainerGMMBase> >("GMMTrainer",
      "This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006", no_init)
  ;

  class_<train::MAP_GMMTrainer, boost::noncopyable, bases<train::GMMTrainer> >("MAP_GMMTrainer",
      "This class implements the maximum a posteriori M-step "
      "of the expectation-maximisation algorithm for a GMM Machine. "
      "The prior parameters are encoded in the form of a GMM (e.g. a universal background model). "
      "The EM algorithm thus performs GMM adaptation.\n"
      "See Section 3.4 of Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000. We use a \"single adaptation coefficient\", alpha_i, and thus a single relevance factor, r.",
      init<optional<double> >((arg("relevance_factor"))))
    .def("setPriorGMM", &train::MAP_GMMTrainer::setPriorGMM, 
      "Set the GMM to use as a prior for MAP adaptation. "
      "Generally, this is a \"universal background model\" (UBM), "
      "also referred to as a \"world model\".")
  ;
 
  class_<train::ML_GMMTrainer, boost::noncopyable, bases<train::GMMTrainer> >("ML_GMMTrainer",
      "This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<optional<bool, bool, bool> >((arg("update_means"), arg("update_variances"), arg("update_weights"))))
  ;

}
