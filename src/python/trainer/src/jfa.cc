/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to Joint Factor Analysis trainers
 */

#include <boost/python.hpp>
#include "trainer/JFATrainer.h"

using namespace boost::python;
namespace train = Torch::trainer;

void bind_trainer_jfa() {
  def("jfa_updateEigen", &train::jfa::updateEigen, (arg("A"), arg("C"), arg("uv")), "Updates eigenchannels (or eigenvoices) from accumulators A and C.");
  def("jfa_estimateXandU", &train::jfa::estimateXandU, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the channel factors.");
  def("jfa_estimateYandV", &train::jfa::estimateYandV, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors y.");
  def("jfa_estimateZandD", &train::jfa::estimateZandD, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors z.");
}
