/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to Joint Factor Analysis trainers
 */

#include <boost/python.hpp>
#include "trainer/JFATrainer.h"
#include "machine/JFAMachine.h"

using namespace boost::python;
namespace train = Torch::trainer;
namespace mach = Torch::machine;

static void jfa_train(train::JFABaseTrainer& t, list list_stats, const size_t n_iter)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const Torch::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const Torch::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const Torch::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.train(gmm_stats, n_iter);
}

static void jfa_train_noinit(train::JFABaseTrainer& t, list list_stats, const size_t n_iter)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const Torch::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const Torch::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const Torch::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.trainNoInit(gmm_stats, n_iter);
}


static void jfa_train_ISV(train::JFABaseTrainer& t, list list_stats, 
  const size_t n_iter, const double relevance_factor)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const Torch::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const Torch::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const Torch::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.trainISV(gmm_stats, n_iter, relevance_factor);
}

static void jfa_train_ISV_noinit(train::JFABaseTrainer& t, list list_stats, 
  const size_t n_iter, const double relevance_factor)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const Torch::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const Torch::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const Torch::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.trainISVNoInit(gmm_stats, n_iter, relevance_factor);
}

static void jfa_enrol(train::JFATrainer& t, list stats, const size_t n_iter)
{
  int n_samples = len(stats);
  std::vector<const Torch::machine::GMMStats*> gmm_stats;
  for(int s=0; s<n_samples; ++s)
    gmm_stats.push_back(extract<const Torch::machine::GMMStats*>(stats[s]));

  // Calls the enrol function
  t.enrol(gmm_stats, n_iter);
}


void bind_trainer_jfa() {
  def("jfa_updateEigen", &train::jfa::updateEigen, (arg("A"), arg("C"), arg("uv")), "Updates eigenchannels (or eigenvoices) from accumulators A and C.");
  def("jfa_estimateXandU", &train::jfa::estimateXandU, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the channel factors.");
  def("jfa_estimateYandV", &train::jfa::estimateYandV, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors y.");
  def("jfa_estimateZandD", &train::jfa::estimateZandD, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors z.");

  class_<train::JFABaseTrainer, boost::noncopyable>("JFABaseTrainer", "Create a trainer for the JFA.", init<mach::JFABaseMachine&>((arg("jfa_base")),"Initializes a new JFABaseTrainer."))
    .add_property("N", make_function(&train::JFABaseTrainer::getN, return_internal_reference<>()), &train::JFABaseTrainer::setN)
    .add_property("F", make_function(&train::JFABaseTrainer::getF, return_internal_reference<>()), &train::JFABaseTrainer::setF)
    .add_property("X", make_function(&train::JFABaseTrainer::getX, return_internal_reference<>()), &train::JFABaseTrainer::setX)
    .add_property("Y", make_function(&train::JFABaseTrainer::getY, return_internal_reference<>()), &train::JFABaseTrainer::setY)
    .add_property("Z", make_function(&train::JFABaseTrainer::getZ, return_internal_reference<>()), &train::JFABaseTrainer::setZ)
    .add_property("VtSigmaInv", make_function(&train::JFABaseTrainer::getVtSigmaInv, return_internal_reference<>()), &train::JFABaseTrainer::setVtSigmaInv)
    .add_property("IdPlusVProd_i", make_function(&train::JFABaseTrainer::getIdPlusVProd_i, return_internal_reference<>()), &train::JFABaseTrainer::setIdPlusVProd_i)
    .add_property("Fn_y_i", make_function(&train::JFABaseTrainer::getFn_y_i, return_internal_reference<>()), &train::JFABaseTrainer::setFn_y_i)
    .add_property("A1_y", make_function(&train::JFABaseTrainer::getA1_y, return_internal_reference<>()), &train::JFABaseTrainer::setA1_y)
    .add_property("A2_y", make_function(&train::JFABaseTrainer::getA2_y, return_internal_reference<>()), &train::JFABaseTrainer::setA2_y)
    .def("setStatistics", &train::JFABaseTrainer::setStatistics, (arg("self"), arg("N"), arg("F")), "Set the zeroth and first order statistics.")
    .def("setSpeakerFactors", &train::JFABaseTrainer::setSpeakerFactors, (arg("self"), arg("x"), arg("y"), arg("z")), "Set the speaker factors.")
    .def("train", (void (train::JFABaseTrainer::*)(const std::vector<blitz::Array<double,2> >&, const std::vector<blitz::Array<double,2> >&, const size_t))&train::JFABaseTrainer::train, (arg("self"), arg("N"), arg("F"), arg("n_iter")), "Call the training procedure.")
    .def("train", &jfa_train, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    .def("trainNoInit", &jfa_train_noinit, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    .def("trainISV", &jfa_train_ISV, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the ISV training procedure.")
    .def("trainISVNoInit", &jfa_train_ISV_noinit, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the ISV training procedure.")
    .def("initializeRandomU", &train::JFABaseTrainer::initializeRandomU, (arg("self")), "Initializes randomly U.")
    .def("initializeRandomV", &train::JFABaseTrainer::initializeRandomV, (arg("self")), "Initializes randomly V.")
    .def("initializeRandomD", &train::JFABaseTrainer::initializeRandomD, (arg("self")), "Initializes randomly D.")
    .def("computeVtSigmaInv", &train::JFABaseTrainer::computeVtSigmaInv, (arg("self")), "Computes Vt*SigmaInv.")
    .def("computeVProd", &train::JFABaseTrainer::computeVProd, (arg("self")), "Computes VProd.")
    .def("computeIdPlusVProd_i", &train::JFABaseTrainer::computeIdPlusVProd_i, (arg("self"), arg("id")), "Computes IdPlusVProd_i.")
    .def("computeFn_y_i", &train::JFABaseTrainer::computeFn_y_i, (arg("self"), arg("id")), "Computes Fn_y_i.")
    .def("updateY_i", &train::JFABaseTrainer::updateY_i, (arg("self"), arg("id")), "Updates Y_i.")
    .def("updateY", &train::JFABaseTrainer::updateY, (arg("self")), "Updates Y.")
    .def("updateV", &train::JFABaseTrainer::updateV, (arg("self")), "Updates V.")
    .def("computeUtSigmaInv", &train::JFABaseTrainer::computeUtSigmaInv, (arg("self")), "Computes Ut*SigmaInv.")
    .def("computeIdPlusUProd_ih", &train::JFABaseTrainer::computeIdPlusUProd_ih, (arg("self"), arg("id"), arg("h")), "Computes IdPlusUProd_ih.")
    .def("computeFn_x_ih", &train::JFABaseTrainer::computeFn_x_ih, (arg("self"), arg("id"), arg("h")), "Computes Fn_x_ih.")
    .def("updateX_ih", &train::JFABaseTrainer::updateX_ih, (arg("self"), arg("id"), arg("h")), "Updates X_ih.")
    .def("updateX", &train::JFABaseTrainer::updateX, (arg("self")), "Updates X.")
    .def("updateU", &train::JFABaseTrainer::updateU, (arg("self")), "Updates U.")
    .def("computeDtSigmaInv", &train::JFABaseTrainer::computeDtSigmaInv, (arg("self")), "Computes Dt*SigmaInv.")
    .def("computeIdPlusDProd_i", &train::JFABaseTrainer::computeIdPlusDProd_i, (arg("self"), arg("id")), "Computes IdPlusDProd_i.")
    .def("computeFn_z_i", &train::JFABaseTrainer::computeFn_z_i, (arg("self"), arg("id")), "Computes Fn_z_i.")
    .def("updateZ_i", &train::JFABaseTrainer::updateZ_i, (arg("self"), arg("id")), "Updates Z_i.")
    .def("updateZ", &train::JFABaseTrainer::updateZ, (arg("self")), "Updates Z.")
    .def("updateD", &train::JFABaseTrainer::updateD, (arg("self")), "Updates D.")
    .def("precomputeSumStatisticsN", &train::JFABaseTrainer::precomputeSumStatisticsN, (arg("self")), "Precomputes zeroth order statistics over sessions.")
    .def("precomputeSumStatisticsF", &train::JFABaseTrainer::precomputeSumStatisticsF, (arg("self")), "Precomputes first order statistics over sessions.")
    ;

  class_<train::JFATrainer, boost::noncopyable>("JFATrainer", "Create a trainer for the JFA.", init<mach::JFAMachine&, train::JFABaseTrainer&>((arg("jfa"), arg("base_trainer")),"Initializes a new JFATrainer."))
    .def("enrol", (void (train::JFATrainer::*)(const blitz::Array<double,2>&, const blitz::Array<double,2>&, const size_t))&train::JFATrainer::enrol, (arg("self"), arg("N"), arg("F"), arg("n_iter")), "Call the training procedure.")
    .def("enrol", &jfa_enrol, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    ;


}
