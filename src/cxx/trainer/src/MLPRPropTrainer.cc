/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 11 Jul 2011 15:59:39 CEST
 *
 * @brief Implementation of the RProp algorithm for MLP training.
 */

#include "math/linear.h"
#include "machine/MLPException.h"
#include "trainer/MLPRPropTrainer.h"

namespace mach = Torch::machine;
namespace math = Torch::math;
namespace train = Torch::trainer;

train::MLPRPropTrainer::MLPRPropTrainer(const MLPStopCriteria& s):
  m_stop(s),
  m_rng()
{
}

train::MLPRPropTrainer::MLPRPropTrainer(size_t max_iterations):
  m_stop(NumberOfIterationsCriteria(max_iterations)),
  m_rng()
{
}

train::MLPRPropTrainer::~MLPRPropTrainer() { }

train::MLPRPropTrainer::MLPRPropTrainer(const MLPRPropTrainer& other):
  m_stop(other.m_stop),
  m_rng()
{
}

train::MLPRPropTrainer& train::MLPRPropTrainer::operator=
(const train::MLPRPropTrainer::MLPRPropTrainer& other) {
  m_stop = other.m_stop;
  m_rng = other.m_rng;
  return *this;
}

/**
 * Constants taken from the paper:
 */
static const double DELTA0 = 0.1;
static const double MIN_STEP = 0.5;
static const double MAX_STEP = 1.2;
static const double DELTA_MAX = 50.0;
static const double DELTA_MIN = 1e-6;

/**
 * A convinient sign function
 */
static uint8_t sign (double x) {
  if (x>0) return +1;
  else if (x<0) return -1;
  return 0;
}

void train::MLPRPropTrainer::train(Torch::machine::MLP& machine,
    const std::vector<Torch::io::Arrayset>& train_data,
    const std::vector<Torch::io::Array>& train_target,
    size_t batch_size) const {

  //The approach here is: we train the network completely outside the machine.
  //By the end, we set the machine and return. We take the network
  //configuration though.

  // (1) Copies all currently set weights from the machine
  const blitz::Array<double,1>& isub = machine.getInputSubraction();
  const blitz::Array<double,1>& idiv = machine.getInputDivision();
  std::vector<blitz::Array<double,2> > weight(machine.numOfHiddenLayers()+1);
  std::vector<blitz::Array<double,1> > bias(machine.numOfHiddenLayers()+1);
 
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();
  for (size_t i=0; i<weight.size(); ++i) {
    weight[i].reference(machine_weight[i].copy());
    bias[i].reference(machine_bias[i].copy());
  }

  // (2) Initializes biases and weights to a random value between -0.1 and 0.1
  boost::uniform_real<double> range(-0.1, 0.1);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >
    urand(m_rng, range);

  for (size_t k=0; k<weight.size(); ++k) {
    for (int i=0; i<weight[k].extent(0); ++i) {
      for (int j=0; j<weight[k].extent(1); ++j) {
        weight[k](i,j) = urand();
      }
    }
    for (int i=0; i<bias[k].extent(0); ++i) {
      bias[k](i) = urand();
    }
  }

  // (2.1) Initializes the activation function (direct and backward)
  mach::MLP::actfun_t actfun = machine.getActivationFunction();

  mach::MLP::actfun_t bwdfun;
  switch (machine.getActivation()) {
    case mach::LINEAR:
      bwdfun = mach::linear_derivative;
      break;
    case mach::TANH:
      bwdfun = mach::tanh_derivative;
      break;
    case mach::LOG:
      bwdfun = mach::logistic_derivative;
      break;
    default:
      throw mach::UnsupportedActivation(machine.getActivation());
  }

  // (3) Pre-allocates the output vector and internal buffers
  // output: values after the activation function
  // target: sampled target values (note on "error" bellow)
  // error: error values; note the matrix is transposed compared to the output
  //        to make it more convinient for the calculations
  blitz::Array<double,2> target(machine.outputSize(), batch_size);
  std::vector<blitz::Array<double,2> > error(machine.numOfHiddenLayers()+1);
  for (size_t i=0; i<error.size(); ++i) {
    error[i].reference(blitz::Array<double,2>(weight[i].extent(1), batch_size));
  }
  //note that "output" will also accomodate the input
  std::vector<blitz::Array<double,2> > output(machine.numOfHiddenLayers()+2);
  output[0].reference(blitz::Array<double,2>(batch_size, machine.inputSize()));
  for (size_t i=1; i<output.size(); ++i) {
    output[i].reference(blitz::Array<double,2>(batch_size, weight[i].extent(1)));
  }

  // There is one generator for each arrayset, since every arrayset can have a
  // different size. 
  std::vector<boost::shared_ptr<boost::variate_generator<boost::mt19937&, boost::uniform_int<size_t> > > > sample_selector(train_data.size());
  for (size_t i=0; i<train_data.size(); ++i) {
    boost::uniform_int<size_t> range(0, train_data[i].size()-1);
    sample_selector[i].reset(new boost::variate_generator<boost::mt19937&, boost::uniform_int<size_t> >(m_rng, range));
  }

  // (5) Training

  // (5.1) Select the samples, copies data to input/target matrix.
  //       Also takes this opportunity to normalize the input.
  size_t counter = 0;
  blitz::Range all = blitz::Range::all();
  while (true) {
    for (size_t i=0; i<train_data.size(); ++i) {
      size_t index = (*sample_selector[i])();
      output[0](counter, all) = 
        (train_data[i].get<double,1>(index) - isub) / idiv;
      target(all, counter) = train_target[i].get<double,1>(); //fixed
      ++counter;
      if (counter >= batch_size) break;
    }
    if (counter >= batch_size) break;
  }

  // (5.2) Forward step -- this is a different implementation than that used on
  // the MLP itself to allow access to some internal buffers.
  for (size_t k=0; k<weight.size(); ++k) { //for all layers
    math::prod_(output[k], weight[k], output[k+1]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<output[k+1].extent(1); ++j) { //for all variables
        output[k+1](i,j) = actfun(output[k+1](i,j) + bias[k](j));
      }
    }
  }

  // (5.3) Calculates the error for the last layer
  error[machine.numOfHiddenLayers()] = target - output.back().transpose(1,0);

  // (5.4) Backward step -- back-propagates the calculated error up to each
  // neuron on the first layer.
  for (size_t k=machine.numOfHiddenLayers(); k>0; --k) {
    math::prod_(weight[k], error[k+1], error[k]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<error[k+1].extent(0); ++j) { //for all variables
        error[k](j,i) *= bwdfun(output[k+1](i,j));
      }
    }
  }

  // (5.5) Weight update -- calculates the weight-update using the RProp rule.
  // This is the place where standard backprop and rprop diverge.
  /*

  data::MeanExtractor mean;
  data::Feature deriv = mean(lesson * input);
  RINGER_DEBUG1("Calculating synaptic weight adjustment with change = "
                << deriv << ", weight update = " << m_weight_update
                << ", previous derivative = " << m_prev_deriv
                << ", previous delta = " << m_prev_delta);

  data::Feature retval = 0;
  int sign_change = SIGN(deriv*m_prev_deriv);

  if (sign_change > 0) {
    m_weight_update = std::min(m_weight_update*MAX_STEP, DELTA_MAX);
  }
  else if (sign_change < 0) {
    m_weight_update = std::max(m_weight_update*MIN_STEP, DELTA_MIN);
    m_prev_deriv = 0;
  }
  else {
    retval = -1 * m_prev_delta;
  }

  if (deriv < 0) retval = -1 * m_weight_update;
  else retval = m_weight_update;
  
  m_prev_deriv = deriv;
  m_prev_delta = retval;
  return retval;
  */
}
