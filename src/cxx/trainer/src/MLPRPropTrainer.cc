/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 11 Jul 2011 15:59:39 CEST
 *
 * @brief Implementation of the RProp algorithm for MLP training.
 */

#include "trainer/MLPRPropTrainer.h"

namespace train = Torch::trainer;

train::MLPRPropTrainer::MLPRPropTrainer(const MLPStopCriteria& s):
  m_stop(s)
{
}

train::MLPRPropTrainer::MLPRPropTrainer(size_t max_iterations):
  m_stop(NumberOfIterationsCriteria(max_iterations))
{
}

train::MLPRPropTrainer::~MLPRPropTrainer() { }

train::MLPRPropTrainer::MLPRPropTrainer(const MLPRPropTrainer& other):
  m_stop(other.m_stop)
{
}

train::MLPRPropTrainer& train::MLPRPropTrainer::operator=
(const train::MLPRPropTrainer::MLPRPropTrainer& other) {
  m_stop = other.m_stop;
  return *this;
}

void train::MLPRPropTrainer::train(Torch::machine::MLP& machine,
    const std::vector<Torch::io::Arrayset>& train_data,
    const std::vector<Torch::io::Array>& train_target,
    size_t batch_size) const {

}
