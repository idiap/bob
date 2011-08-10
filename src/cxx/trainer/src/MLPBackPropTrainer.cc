/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 11 Jul 2011 15:59:39 CEST
 *
 * @brief Implementation of the BackProp algorithm for MLP training.
 */

#include <algorithm>
#include "core/array_check.h"
#include "math/linear.h"
#include "machine/MLPException.h"
#include "trainer/Exception.h"
#include "trainer/MLPBackPropTrainer.h"

namespace array = Torch::core::array;
namespace mach = Torch::machine;
namespace math = Torch::math;
namespace train = Torch::trainer;

train::MLPBackPropTrainer::MLPBackPropTrainer(const mach::MLP& machine,
    size_t batch_size):
  m_learning_rate(0.1),
  m_momentum(0.0),
  m_train_bias(true),
  m_H(machine.numOfHiddenLayers()), ///< handy!
  m_weight_ref(m_H + 1),
  m_bias_ref(m_H + 1),
  m_delta(m_H + 1),
  m_delta_bias(m_H + 1),
  m_prev_delta(m_H + 1),
  m_prev_delta_bias(m_H + 1),
  m_actfun(machine.getActivationFunction()),
  m_bwdfun(),
  m_target(),
  m_error(m_H + 1),
  m_output(m_H + 2)
{
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_delta[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_delta_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
    m_prev_delta[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_prev_delta_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
  }

  reset();

  switch (machine.getActivation()) {
    case mach::LINEAR:
      m_bwdfun = mach::linear_derivative;
      break;
    case mach::TANH:
      m_bwdfun = mach::tanh_derivative;
      break;
    case mach::LOG:
      m_bwdfun = mach::logistic_derivative;
      break;
    default:
      throw mach::UnsupportedActivation(machine.getActivation());
  }

  setBatchSize(batch_size);
}

train::MLPBackPropTrainer::~MLPBackPropTrainer() { }

train::MLPBackPropTrainer::MLPBackPropTrainer(const MLPBackPropTrainer& other):
  m_learning_rate(other.m_learning_rate),
  m_momentum(other.m_momentum),
  m_train_bias(other.m_train_bias),
  m_H(other.m_H),
  m_weight_ref(m_H + 1),
  m_bias_ref(m_H + 1),
  m_delta(m_H + 1),
  m_delta_bias(m_H + 1),
  m_prev_delta(m_H + 1),
  m_prev_delta_bias(m_H + 1),
  m_actfun(other.m_actfun),
  m_bwdfun(other.m_bwdfun),
  m_target(Torch::core::array::ccopy(other.m_target)),
  m_error(m_H + 1),
  m_output(m_H + 2)
{
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_delta[k].reference(Torch::core::array::ccopy(other.m_delta[k]));
    m_delta_bias[k].reference(Torch::core::array::ccopy(other.m_delta_bias[k]));
    m_prev_delta[k].reference(Torch::core::array::ccopy(other.m_prev_delta[k]));
    m_prev_delta_bias[k].reference(Torch::core::array::ccopy(other.m_prev_delta_bias[k]));
    m_error[k].reference(Torch::core::array::ccopy(other.m_error[k]));
    m_output[k].reference(Torch::core::array::ccopy(other.m_output[k]));
  }
  m_output[m_H + 1].reference(Torch::core::array::ccopy(other.m_output[m_H + 1]));
}

train::MLPBackPropTrainer& train::MLPBackPropTrainer::operator=
(const train::MLPBackPropTrainer::MLPBackPropTrainer& other) {
  m_learning_rate = other.m_learning_rate;
  m_momentum = other.m_momentum;
  m_train_bias = other.m_train_bias;
  m_H = other.m_H;
  m_weight_ref.resize(m_H + 1);
  m_bias_ref.resize(m_H + 1);
  m_delta.resize(m_H + 1);
  m_delta_bias.resize(m_H + 1);
  m_prev_delta.resize(m_H + 1);
  m_prev_delta_bias.resize(m_H + 1);
  m_actfun = other.m_actfun;
  m_bwdfun = other.m_bwdfun;
  m_target.reference(Torch::core::array::ccopy(other.m_target));
  m_error.resize(m_H + 1);
  m_output.resize(m_H + 2);

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_delta[k].reference(Torch::core::array::ccopy(other.m_delta[k]));
    m_delta_bias[k].reference(Torch::core::array::ccopy(other.m_delta_bias[k]));
    m_prev_delta[k].reference(Torch::core::array::ccopy(other.m_prev_delta[k]));
    m_prev_delta_bias[k].reference(Torch::core::array::ccopy(other.m_prev_delta_bias[k]));
    m_error[k].reference(Torch::core::array::ccopy(other.m_error[k]));
    m_output[k].reference(Torch::core::array::ccopy(other.m_output[k]));
  }
  m_output[m_H + 1].reference(Torch::core::array::ccopy(other.m_output[m_H + 1]));

  return *this;
}

void train::MLPBackPropTrainer::reset() {
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_prev_delta[k] = 0;
    m_prev_delta_bias[k] = 0;
  }
}

void train::MLPBackPropTrainer::setBatchSize (size_t batch_size) {
  // m_output: values after the activation function; note that "output" will
  //           accomodate the input to ease on the calculations
  // m_target: sampled target values
  // m_error: error values;
  
  m_target.resize(batch_size, m_delta.back().extent(1));

  m_output[0].resize(batch_size, m_delta[0].extent(0));

  for (size_t k=1; k<m_output.size(); ++k) {
    m_output[k].resize(batch_size, m_delta[k-1].extent(1));
  }

  for (size_t k=0; k<m_error.size(); ++k) {
    m_error[k].resize(batch_size, m_delta[k].extent(1));
  }
}

bool train::MLPBackPropTrainer::isCompatible(const mach::MLP& machine) const 
{
  if (m_H != machine.numOfHiddenLayers()) return false;
  
  if (m_target.extent(1) != (int)machine.outputSize()) return false;

  if (m_output[0].extent(1) != (int)machine.inputSize()) return false;

  //also, each layer should be of the same size
  for (size_t k=0; k<(m_H + 1); ++k) {
    if (!array::hasSameShape(m_delta[k], machine.getWeights()[k])) return false;
  }

  //if you get to this point, you can only return true
  return true;
}

void train::MLPBackPropTrainer::forward_step() {
  size_t batch_size = m_target.extent(0);
  for (size_t k=0; k<m_weight_ref.size(); ++k) { //for all layers
    math::prod_(m_output[k], m_weight_ref[k], m_output[k+1]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<m_output[k+1].extent(1); ++j) { //for all variables
        m_output[k+1](i,j) = m_actfun(m_output[k+1](i,j) + m_bias_ref[k](j));
      }
    }
  }
}

void train::MLPBackPropTrainer::backward_step() {
  size_t batch_size = m_target.extent(0);
  //last layer
  m_error[m_H] = m_target - m_output.back();
  for (int i=0; i<(int)batch_size; ++i) { //for every example
    for (int j=0; j<m_error[m_H].extent(1); ++j) { //for all variables
      m_error[m_H](i,j) *= m_bwdfun(m_output[m_H+1](i,j));
    }
  }

  //all other layers
  for (size_t k=m_H; k>0; --k) {
    math::prod_(m_error[k], m_weight_ref[k].transpose(1,0), m_error[k-1]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<m_error[k-1].extent(1); ++j) { //for all variables
        m_error[k-1](i,j) *= m_bwdfun(m_output[k](i,j));
      }
    }
  }
}

void train::MLPBackPropTrainer::backprop_weight_update() {
  size_t batch_size = m_target.extent(0);
  for (size_t k=0; k<m_weight_ref.size(); ++k) { //for all layers
    math::prod_(m_output[k].transpose(1,0), m_error[k], m_delta[k]);
    m_delta[k] *= m_learning_rate / batch_size;
    m_weight_ref[k] += ((1-m_momentum)*m_delta[k]) + 
      (m_momentum*m_prev_delta[k]);
    m_prev_delta[k] = m_delta[k];

    // Here we decide if we should train the biases or not
    if (!m_train_bias) continue;

    // We do the same for the biases, with the exception that biases can be
    // considered as input neurons connecting the respective layers, with a
    // fixed input = +1. This means we only need to probe for the error at
    // layer k.
    blitz::secondIndex J;
    m_delta_bias[k] = m_learning_rate * 
      blitz::mean(m_error[k].transpose(1,0), J);
    m_bias_ref[k] += ((1-m_momentum)*m_delta_bias[k]) + 
      (m_momentum*m_prev_delta_bias[k]);
    m_prev_delta_bias[k] = m_delta_bias[k];
  }
}

void train::MLPBackPropTrainer::train(Torch::machine::MLP& machine,
    const blitz::Array<double,2>& input,
    const blitz::Array<double,2>& target) {
  if (!isCompatible(machine)) throw train::IncompatibleMachine();
  array::assertSameDimensionLength(getBatchSize(), input.extent(0));
  array::assertSameDimensionLength(getBatchSize(), target.extent(0));
  train_(machine, input, target);
}

void train::MLPBackPropTrainer::train_(Torch::machine::MLP& machine,
    const blitz::Array<double,2>& input,
    const blitz::Array<double,2>& target) {

  m_output[0].reference(input);
  m_target.reference(target);

  // We refer to the machine's weights and biases
  for (size_t k=0;k<m_weight_ref.size();++k)
    m_weight_ref[k].reference(machine.getWeights()[k]);
  for (size_t k=0;k<m_bias_ref.size();++k)
    m_bias_ref[k].reference(machine.getBiases()[k]);

  // To be called in this sequence for a general backprop algorithm
  forward_step();
  backward_step();
  backprop_weight_update();
}
