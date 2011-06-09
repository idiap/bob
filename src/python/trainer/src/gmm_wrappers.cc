#include <boost/python.hpp>
#include "database/Arrayset.h"
#include "trainer/GMMTrainer.h"
#include "trainer/MAP_GMMTrainer.h"
#include "trainer/ML_GMMTrainer.h"

using namespace boost::python;
namespace train = Torch::trainer;
namespace mach = Torch::machine;
namespace db = Torch::database;


class EMTrainerGMMWrapper: public train::EMTrainer<mach::GMMMachine, db::Arrayset>, 
                           public wrapper<train::EMTrainer<mach::GMMMachine, db::Arrayset> > 
{
public:
  EMTrainerGMMWrapper(double convergence_threshold = 0.001, int max_iterations = 10):
    train::EMTrainer<mach::GMMMachine, db::Arrayset >(convergence_threshold, max_iterations) {}

  virtual ~EMTrainerGMMWrapper() {}
 
  void initialization(mach::GMMMachine& machine, const db::Arrayset& data) {
    this->get_override("initialization")(machine, data);
  }
  
  double eStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    return this->get_override("eStep")(machine, data);
  }
  
  void mStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    this->get_override("mStep")(machine, data);
  }

  void train(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::EMTrainer<mach::GMMMachine, db::Arrayset>::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::EMTrainer<mach::GMMMachine, db::Arrayset>::train(machine, data);
  }

};


class GMMTrainerWrapper: public train::GMMTrainer,
                         public wrapper<train::GMMTrainer>
{
public:
  GMMTrainerWrapper(bool update_means = true, bool update_variances = false, bool update_weights = false):
    train::GMMTrainer(update_means, update_variances, update_weights) {}

  virtual ~GMMTrainerWrapper() {}
  
  void initialization(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_initialization = this->get_override("initialization")) 
      python_initialization(machine, data);
    else
      train::GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::GMMTrainer::initialization(machine, data);
  }
  
  double eStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_eStep = this->get_override("eStep")) return python_eStep(machine, data);
    return train::GMMTrainer::eStep(machine, data);
  }
  
  double d_eStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    return train::GMMTrainer::eStep(machine, data);
  }

  void mStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    this->get_override("mStep")(machine, data);
  }

  void train(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::GMMTrainer::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::GMMTrainer::train(machine, data);
  }
 
};


class MAP_GMMTrainerWrapper: public train::MAP_GMMTrainer,
                             public wrapper<train::MAP_GMMTrainer>
{
public:
  MAP_GMMTrainerWrapper(double relevance_factor = 0):
    train::MAP_GMMTrainer(relevance_factor) {}

  virtual ~MAP_GMMTrainerWrapper() {}

  void initialization(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_initialization = this->get_override("initialization")) 
      python_initialization(machine, data);
    else
      train::MAP_GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::MAP_GMMTrainer::initialization(machine, data);
  }
  
  double eStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_eStep = this->get_override("eStep")) return python_eStep(machine, data);
    return train::MAP_GMMTrainer::eStep(machine, data);
  }
  
  double d_eStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    return train::MAP_GMMTrainer::eStep(machine, data);
  }

  void mStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_mStep = this->get_override("mStep")) 
      python_mStep(machine, data);
    else
      train::MAP_GMMTrainer::mStep(machine, data);
  }
  
  void d_mStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::MAP_GMMTrainer::mStep(machine, data);
  }

  void train(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::MAP_GMMTrainer::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::MAP_GMMTrainer::train(machine, data);
  }
 
};

class ML_GMMTrainerWrapper: public train::ML_GMMTrainer,
                            public wrapper<train::ML_GMMTrainer>
{
public:
  ML_GMMTrainerWrapper(bool update_means = true, bool update_variances = false, bool update_weights = false):
    train::ML_GMMTrainer(update_means, update_variances, update_weights) {}

  virtual ~ML_GMMTrainerWrapper() {}

  void initialization(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_initialization = this->get_override("initialization")) 
      python_initialization(machine, data);
    else
      train::ML_GMMTrainer::initialization(machine, data);
  }
  
  void d_initialization(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::ML_GMMTrainer::initialization(machine, data);
  }
  
  double eStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_eStep = this->get_override("eStep")) return python_eStep(machine, data);
    return train::ML_GMMTrainer::eStep(machine, data);
  }
  
  double d_eStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    return train::ML_GMMTrainer::eStep(machine, data);
  }

  void mStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_mStep = this->get_override("mStep")) 
      python_mStep(machine, data);
    else
      train::ML_GMMTrainer::mStep(machine, data);
  }
  
  void d_mStep(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::ML_GMMTrainer::mStep(machine, data);
  }

  void train(mach::GMMMachine& machine, const db::Arrayset& data) {
    if (override python_train = this->get_override("train")) 
      python_train(machine, data);
    else
      train::ML_GMMTrainer::train(machine, data);
  }

  void d_train(mach::GMMMachine& machine, const db::Arrayset& data) {
    train::ML_GMMTrainer::train(machine, data);
  }
 
};


void bind_trainer_gmm_wrappers() {

  typedef train::EMTrainer<mach::GMMMachine, db::Arrayset> EMTrainerGMMBase; 

  class_<EMTrainerGMMWrapper, boost::noncopyable >("EMTrainerGMM", no_init)
    .def(init<optional<double, int> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10)))
    .add_property("convergenceThreshold", &EMTrainerGMMBase::getConvergenceThreshold, &EMTrainerGMMBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &EMTrainerGMMBase::getMaxIterations, &EMTrainerGMMBase::setMaxIterations, "Max iterations")
    .def("train", &EMTrainerGMMBase::train, &EMTrainerGMMWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using data")
    .def("initialization", pure_virtual(&EMTrainerGMMBase::initialization), (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", pure_virtual(&EMTrainerGMMBase::eStep), (arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("mStep", pure_virtual(&EMTrainerGMMBase::mStep), (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
  ;


  class_<GMMTrainerWrapper, boost::noncopyable >("GMMTrainer",
      "This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<optional<bool, bool, bool> >((arg("update_means"), arg("update_variances"), arg("update_weights"))))
    .add_property("convergenceThreshold", &train::GMMTrainer::getConvergenceThreshold, &train::GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &train::GMMTrainer::getMaxIterations, &train::GMMTrainer::setMaxIterations, "Max iterations")
    .def("train", &train::GMMTrainer::train, &GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &train::GMMTrainer::initialization, &GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", &train::GMMTrainer::eStep, &GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("mStep", pure_virtual(&train::GMMTrainer::mStep), (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;


  class_<MAP_GMMTrainerWrapper, boost::noncopyable >("MAP_GMMTrainer",
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
    .add_property("convergenceThreshold", &train::MAP_GMMTrainer::getConvergenceThreshold, &train::MAP_GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &train::MAP_GMMTrainer::getMaxIterations, &train::MAP_GMMTrainer::setMaxIterations, "Max iterations")
    .def("train", &train::MAP_GMMTrainer::train, &MAP_GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &train::MAP_GMMTrainer::initialization, &MAP_GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", &train::MAP_GMMTrainer::eStep, &MAP_GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("mStep", &train::MAP_GMMTrainer::mStep, &MAP_GMMTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;

   
  class_<ML_GMMTrainerWrapper, boost::noncopyable >("ML_GMMTrainer",
      "This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.\n"
      "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<optional<bool, bool, bool> >((arg("update_means"), arg("update_variances"), arg("update_weights"))))
    .add_property("convergenceThreshold", &train::ML_GMMTrainer::getConvergenceThreshold, &train::ML_GMMTrainer::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &train::ML_GMMTrainer::getMaxIterations, &train::ML_GMMTrainer::setMaxIterations, "Max iterations")
    .def("train", &train::ML_GMMTrainer::train, &ML_GMMTrainerWrapper::d_train, (arg("machine"), arg("data")), "Train a machine using some data")
    .def("initialization", &train::ML_GMMTrainer::initialization, &ML_GMMTrainerWrapper::d_initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("eStep", &train::ML_GMMTrainer::eStep, &ML_GMMTrainerWrapper::d_eStep, (arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("mStep", &train::ML_GMMTrainer::mStep, &ML_GMMTrainerWrapper::d_mStep, (arg("machine"), arg("data")), "M-step of the EM-algorithm.")
  ;

}
