/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 11 Jul 2011 15:59:39 CEST
 *
 * @brief Implementation of the RProp algorithm for MLP training.
 */

#include <algorithm>
#include "math/linear.h"
#include "machine/MLPException.h"
#include "trainer/MLPRPropTrainer.h"

namespace mach = Torch::machine;
namespace math = Torch::math;
namespace train = Torch::trainer;

/**
 * Constants taken from the paper:
 */
static const double DELTA0 = 0.1;
static const double ETA_MINUS = 0.5;
static const double ETA_PLUS = 1.2;
static const double DELTA_MAX = 50.0;
static const double DELTA_MIN = 1e-6;

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
 * A function that returns the sign of a double number (zero if the value is
 * 0).
 */
static int8_t sign (double x) {
  if (x > 0) return +1;
  return (x == 0)? 0 : -1;
}

void train::MLPRPropTrainer::train(Torch::machine::MLP& machine,
    const std::vector<Torch::io::Arrayset>& train_data,
    const std::vector<Torch::io::Array>& train_target,
    size_t batch_size) const {

  //The approach here is: we train the network completely outside the machine.
  //By the end, we set the machine and return. We take the network
  //configuration though.

  const size_t H = machine.numOfHiddenLayers(); //handy

  // (1) Copies all currently set weights from the machine, sets up a few extra
  // arrays to cache data during the calculations
  const blitz::Array<double,1>& isub = machine.getInputSubraction();
  const blitz::Array<double,1>& idiv = machine.getInputDivision();
  
  std::vector<blitz::Array<double,2> > weight(H+1);
  std::vector<blitz::Array<double,1> > bias(H+1);

  std::vector<blitz::Array<double,2> > delta(H+1);
  std::vector<blitz::Array<double,1> > delta_bias(H+1);
  
  std::vector<blitz::Array<double,2> > deriv(H+1);
  std::vector<blitz::Array<double,1> > deriv_bias(H+1);
  
  std::vector<blitz::Array<double,2> > prev_deriv(H+1);
  std::vector<blitz::Array<double,1> > prev_deriv_bias(H+1);
 
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  for (size_t k=0; k<weight.size(); ++k) {
    weight[k].reference(machine_weight[k].copy());
    bias[k].reference(machine_bias[k].copy());
    deriv[k].reference(blitz::Array<double,2>(weight[k].shape()));
    deriv_bias[k].reference(blitz::Array<double,1>(bias[k].shape()));
    delta[k].reference(blitz::Array<double,2>(weight[k].shape()));
    delta[k] = DELTA0; ///< fixed initialization as proposed by RProp
    delta_bias[k].reference(blitz::Array<double,1>(bias[k].shape()));
    delta_bias[k] = DELTA0;
    prev_deriv[k].reference(blitz::Array<double,2>(weight[k].shape()));
    prev_deriv[k] = 0;
    prev_deriv_bias[k].reference(blitz::Array<double,1>(bias[k].shape()));
    prev_deriv_bias[k] = 0;
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
  std::vector<blitz::Array<double,2> > error(H+1);
  for (size_t i=0; i<error.size(); ++i) {
    error[i].reference(blitz::Array<double,2>(weight[i].extent(1), batch_size));
  }
  //note that "output" will also accomodate the input
  std::vector<blitz::Array<double,2> > output(H+2);
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

  // (5.2) Forward step -- this is a second implementation of that used on
  // the MLP itself to allow access to some internal buffers. In our current
  // setup, we keep the "output"'s of every individual layer separately as we
  // are going to need them for the weight update.
  for (size_t k=0; k<weight.size(); ++k) { //for all layers
    math::prod_(output[k], weight[k], output[k+1]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<output[k+1].extent(1); ++j) { //for all variables
        output[k+1](i,j) = actfun(output[k+1](i,j) + bias[k](j));
      }
    }
  }

  // (5.3) Calculates the error for the last layer
  //       At this point we take the opportunity to back-propagate the error to
  //       "before" the last neuron layer.
  error[H] = target - output.back().transpose(1,0);
  for (int i=0; i<(int)batch_size; ++i) { //for every example
    for (int j=0; j<error[H+1].extent(0); ++j) { //for all variables
      error[H](j,i) *= bwdfun(output[H+1](i,j));
    }
  }

  // (5.4) Backward step -- back-propagates the calculated error up to each
  // neuron on the first layer. This is explained on Bishop's formula 5.55 and
  // 5.56, at page 244 (see also figure 5.7 for a graphical representation).
  for (size_t k=H; k>0; --k) {
    math::prod_(weight[k], error[k+1], error[k]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<error[k+1].extent(0); ++j) { //for all variables
        error[k](j,i) *= bwdfun(output[k+1](i,j));
      }
    }
  }

  // (5.5) Weight update -- calculates the weight-update using derivatives
  //       as explained in Bishop's formula 5.53, page 243.

  // Note: For RProp, specifically, we only care about the derivative's sign,
  // current and the previous. This is the place where standard backprop and
  // rprop diverge.
  //
  // For extra insight, double-check the Technical Report entitled "Rprop -
  // Description and Implementation Details" by Martin Riedmiller, 1994. Just
  // browse the internet for it. Keep it under your pillow ;-)

  for (size_t k=0; k<weight.size(); ++k) { //for all layers
    math::prod_(output[k].transpose(1,0), error[k].transpose(1,0), deriv[k]);

    // Note that we don't need to estimate the mean since we are only
    // interested in the sign of the derivative and dividing by the mean makes
    // no difference on the final result as 'batch_size' is always > 0!
    // deriv[k] /= batch_size; //estimates the mean for the batch

    // Calculates the sign change as prescribed on the RProp paper. Depending
    // on the sign change, we update the "weight_update" matrix and apply the
    // updates on the respective weights.
    for (int i=0; i<deriv[k].extent(0); ++i) {
      for (int j=0; j<deriv[k].extent(1); ++j) {
        int8_t M = sign(deriv[k](i,j) * prev_deriv[k](i,j));
        // Implementations equations (4-6) on the RProp paper:
        if (M > 0) {
          delta[k](i,j) = std::min(delta[k](i,j)*ETA_PLUS, DELTA_MAX); 
          weight[k](i,j) -= sign(deriv[k](i,j)) * delta[k](i,j); 
          prev_deriv[k](i,j) = deriv[k](i,j);
        }
        else if (M < 0) {
          delta[k](i,j) = std::max(delta[k](i,j)*ETA_MINUS, DELTA_MIN);
          prev_deriv[k](i,j) = 0;
        }
        else { //M == 0
          weight[k](i,j) -= sign(deriv[k](i,j)) * delta[k](i,j);
        }
      }
    }

    // We do the same for the biases, with the exception that biases can be
    // considered as input neurons connecting the respective layers, with a
    // fixed input = +1. This means we only need to probe for the error at
    // layer k.
    blitz::secondIndex J;
    deriv_bias[k] = blitz::sum(error[k], J);
    for (int i=0; i<deriv_bias[k].extent(0); ++i) {
      int8_t M = sign(deriv_bias[k](i));
      // Implementations equations (4-6) on the RProp paper:
      if (M > 0) {
        delta_bias[k](i) = std::min(delta_bias[k](i)*ETA_PLUS, DELTA_MAX); 
        bias[k](i) -= sign(deriv_bias[k](i)) * delta_bias[k](i); 
        prev_deriv_bias[k](i) = deriv_bias[k](i);
      }
      else if (M < 0) {
        delta_bias[k](i) = std::max(delta_bias[k](i)*ETA_MINUS, DELTA_MIN);
        prev_deriv_bias[k](i) = 0;
      }
      else { //M == 0
        bias[k](i) -= sign(deriv_bias[k](i)) * delta_bias[k](i);
      }
    }
  }

  // (6) Machine updating and that's all, folks
  machine.setWeights(weight);
  machine.setBiases(bias);
}
