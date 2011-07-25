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

void bind_trainer_jfa() {
  def("jfa_updateEigen", &train::jfa::updateEigen, (arg("A"), arg("C"), arg("uv")), "Updates eigenchannels (or eigenvoices) from accumulators A and C.");
  def("jfa_estimateXandU", &train::jfa::estimateXandU, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the channel factors.");
  def("jfa_estimateYandV", &train::jfa::estimateYandV, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors y.");
  def("jfa_estimateZandD", &train::jfa::estimateZandD, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors z.");

  class_<train::JFATrainer, boost::noncopyable>("JFATrainer", "Create a trainer for the JFA.", init<mach::JFAMachine&>((arg("jfa_machine")),"Initializes a new JFATrainer."))
    .add_property("N", make_function(&train::JFATrainer::getN, return_internal_reference<>()), &train::JFATrainer::setN)
    .add_property("F", make_function(&train::JFATrainer::getF, return_internal_reference<>()), &train::JFATrainer::setF)
    .add_property("X", make_function(&train::JFATrainer::getX, return_internal_reference<>()), &train::JFATrainer::setX)
    .add_property("Y", make_function(&train::JFATrainer::getY, return_internal_reference<>()), &train::JFATrainer::setY)
    .add_property("Z", make_function(&train::JFATrainer::getZ, return_internal_reference<>()), &train::JFATrainer::setZ)
    .add_property("VtSigmaInv", make_function(&train::JFATrainer::getVtSigmaInv, return_internal_reference<>()), &train::JFATrainer::setVtSigmaInv)
    .add_property("IdPlusVProd_i", make_function(&train::JFATrainer::getIdPlusVProd_i, return_internal_reference<>()), &train::JFATrainer::setIdPlusVProd_i)
    .add_property("Fn_y_i", make_function(&train::JFATrainer::getFn_y_i, return_internal_reference<>()), &train::JFATrainer::setFn_y_i)
    .add_property("A1_y", make_function(&train::JFATrainer::getA1_y, return_internal_reference<>()), &train::JFATrainer::setA1_y)
    .add_property("A2_y", make_function(&train::JFATrainer::getA2_y, return_internal_reference<>()), &train::JFATrainer::setA2_y)
    .def("setStatistics", &train::JFATrainer::setStatistics, (arg("self"), arg("N"), arg("F")), "Set the zeroth and first order statistics.")
    .def("setSpeakerFactors", &train::JFATrainer::setSpeakerFactors, (arg("self"), arg("x"), arg("y"), arg("z")), "Set the speaker factors.")
    .def("initializeRandomU", &train::JFATrainer::initializeRandomU, (arg("self")), "Initializes randomly U.")
    .def("initializeRandomV", &train::JFATrainer::initializeRandomV, (arg("self")), "Initializes randomly V.")
    .def("initializeRandomD", &train::JFATrainer::initializeRandomD, (arg("self")), "Initializes randomly D.")
    .def("computeVtSigmaInv", &train::JFATrainer::computeVtSigmaInv, (arg("self")), "Computes Vt*SigmaInv.")
    .def("computeVProd", &train::JFATrainer::computeVProd, (arg("self")), "Computes VProd.")
    .def("computeIdPlusVProd_i", &train::JFATrainer::computeIdPlusVProd_i, (arg("self"), arg("id")), "Computes IdPlusVProd_i.")
    .def("computeFn_y_i", &train::JFATrainer::computeFn_y_i, (arg("self"), arg("id")), "Computes Fn_y_i.")
    .def("updateY_i", &train::JFATrainer::updateY_i, (arg("self"), arg("id")), "Updates Y_i.")
    .def("updateY", &train::JFATrainer::updateY, (arg("self")), "Updates Y.")
    .def("updateV", &train::JFATrainer::updateV, (arg("self")), "Updates V.")
    .def("computeUtSigmaInv", &train::JFATrainer::computeUtSigmaInv, (arg("self")), "Computes Ut*SigmaInv.")
    .def("computeIdPlusUProd_ih", &train::JFATrainer::computeIdPlusUProd_ih, (arg("self"), arg("id"), arg("h")), "Computes IdPlusUProd_ih.")
    .def("computeFn_x_ih", &train::JFATrainer::computeFn_x_ih, (arg("self"), arg("id"), arg("h")), "Computes Fn_x_ih.")
    .def("updateX_ih", &train::JFATrainer::updateX_ih, (arg("self"), arg("id"), arg("h")), "Updates X_ih.")
    .def("updateX", &train::JFATrainer::updateX, (arg("self")), "Updates X.")
    .def("updateU", &train::JFATrainer::updateU, (arg("self")), "Updates U.")
    .def("computeDtSigmaInv", &train::JFATrainer::computeDtSigmaInv, (arg("self")), "Computes Dt*SigmaInv.")
    .def("computeIdPlusDProd_i", &train::JFATrainer::computeIdPlusDProd_i, (arg("self"), arg("id")), "Computes IdPlusDProd_i.")
    .def("computeFn_z_i", &train::JFATrainer::computeFn_z_i, (arg("self"), arg("id")), "Computes Fn_z_i.")
    .def("updateZ_i", &train::JFATrainer::updateZ_i, (arg("self"), arg("id")), "Updates Z_i.")
    .def("updateZ", &train::JFATrainer::updateZ, (arg("self")), "Updates Z.")
    .def("updateD", &train::JFATrainer::updateD, (arg("self")), "Updates D.")
    .def("precomputeSumStatisticsN", &train::JFATrainer::precomputeSumStatisticsN, (arg("self")), "Precomputes zeroth order statistics over sessions.")
    .def("precomputeSumStatisticsF", &train::JFATrainer::precomputeSumStatisticsF, (arg("self")), "Precomputes first order statistics over sessions.")
    ;

}
