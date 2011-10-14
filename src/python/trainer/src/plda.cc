/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to Probabilistic Linear Discriminant Analysis 
 * trainers.
 */

#include <boost/python.hpp>
#include "io/Arrayset.h"
#include "machine/PLDAMachine.h"
#include "trainer/PLDATrainer.h"

using namespace boost::python;
namespace train = Torch::trainer;
namespace mach = Torch::machine;
namespace io = Torch::io;


static void plda_train(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<io::Arrayset> v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    io::Arrayset ar = extract<io::Arrayset>(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the train function
  t.train(m, v_arraysets);
}

static void plda_initialization(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<io::Arrayset> v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    io::Arrayset ar = extract<io::Arrayset>(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the initialization function
  t.initialization(m, v_arraysets);
}

static void plda_eStep(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<io::Arrayset> v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    io::Arrayset ar = extract<io::Arrayset>(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the eStep function
  t.eStep(m, v_arraysets);
}

static void plda_mStep(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<io::Arrayset> v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    io::Arrayset ar = extract<io::Arrayset>(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the mStep function
  t.mStep(m, v_arraysets);
}

static void plda_finalization(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<io::Arrayset> v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    io::Arrayset ar = extract<io::Arrayset>(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the finalization function
  t.finalization(m, v_arraysets);
}



void bind_trainer_plda() {
  typedef train::EMTrainerNew<mach::PLDABaseMachine, std::vector<io::Arrayset> > EMTrainerPLDABase; 

  class_<EMTrainerPLDABase, boost::noncopyable>("EMTrainerPLDA", "The base python class for all EM/PLDA-based trainers.", no_init)
    .add_property("convergenceThreshold", &EMTrainerPLDABase::getConvergenceThreshold, &EMTrainerPLDABase::setConvergenceThreshold, "Convergence threshold")
    .add_property("maxIterations", &EMTrainerPLDABase::getMaxIterations, &EMTrainerPLDABase::setMaxIterations, "Max iterations")
    .add_property("computeLikelihoodVariable", &EMTrainerPLDABase::getComputeLikelihood, &EMTrainerPLDABase::setComputeLikelihood, "Indicates whether the log likelihood should be computed during EM or not")
    .def("train", &EMTrainerPLDABase::train, (arg("machine"), arg("data")), "Trains a machine using data")
    .def("initialization", &EMTrainerPLDABase::initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &EMTrainerPLDABase::finalization, (arg("machine"), arg("data")), "This method is called at the end of the EM algorithm")
    .def("eStep", &EMTrainerPLDABase::eStep, (arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("mStep", &EMTrainerPLDABase::mStep, (arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("computeLikelihood", &EMTrainerPLDABase::computeLikelihood, (arg("machine"), arg("data")), "Computes the current log likelihood given the hidden variable distribution (or the sufficient statistics)")
  ;


  class_<train::PLDABaseTrainer, boost::noncopyable, bases<EMTrainerPLDABase> >("PLDABaseTrainer", "Creates a trainer for a PLDABaseMachine.", init<int, int, optional<double,double,bool> >((arg("nf"), arg("ng"), arg("convergence_threshold"), arg("max_iterations"), arg("compute_likelihood")),"Initializes a new PLDABaseTrainer."))
    .add_property("z_first_order", make_function(&train::PLDABaseTrainer::getZFirstOrder, return_internal_reference<>()))
    .add_property("z_second_order_sum", make_function(&train::PLDABaseTrainer::getZSecondOrderSum, return_internal_reference<>()))
    .def("train", &plda_train, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the training procedure. This will call initialization(), a loop of eStep() and mStep(), and finalization().")
    .def("initialization", &plda_initialization, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the initialization method of the training procedure.")
    .def("eStep", &plda_eStep, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the eStep method of the training procedure.")
    .def("mStep", &plda_mStep, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the mStep method of the training procedure.")
    .def("finalization", &plda_finalization, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the finalization method of the training procedure.")
    ;
}
