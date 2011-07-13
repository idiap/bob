/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 13 Jul 2011 17:08:34 CEST
 *
 * @brief Python bindings for RProp training
 */

#include <boost/python.hpp>
#include "trainer/DataShuffler.h"
#include "trainer/MLPRPropTrainer.h"

using namespace boost::python;
namespace io = Torch::io;
namespace mach = Torch::machine;
namespace train = Torch::trainer;

static void random1(const train::MLPRPropTrainer& T, Torch::machine::MLP& M) {
  T.random(M);
}

static void random2(const train::MLPRPropTrainer& T, Torch::machine::MLP& M,
   double lower_bound, double upper_bound) {
  T.random(M, lower_bound, upper_bound);
}

void bind_trainer_rprop() {
  class_<train::DataShuffler>("DataShuffler", "A data shuffler is capable of being populated with data from one or multiple classes and matching target values. Once setup, the shuffer can randomly select a number of vectors and accompaning targets for the different classes, filling up user containers.\n\nData shufflers are particular useful for training neural networks.", init<const std::vector<io::Arrayset>&, const std::vector<blitz::Array<double,1> >&>((arg("data"), arg("target")), "Initializes the shuffler with some data classes and corresponding targets. Note that Arraysets must have (for the time being), a shape of (1,) and an element type == double."))
    .def("setSeed", &train::DataShuffler::setSeed, (arg("self"), arg("seed")), "Sets the seed of the internal random number generator. We actually keep as many random generators as classes so we will set the first generator with the seed you give and the next ones with an incremented seed value until all generators have had their seeds set.")
    .def("stdnorm", &train::DataShuffler::getStdNorm, (arg("self"), arg("mean"), arg("stddev")), "Calculates and returns mean and standard deviation from the input data.")
    .add_property("auto_stdnorm", &train::DataShuffler::getAutoStdNorm, &train::DataShuffler::setAutoStdNorm)
    .def("__call__", &train::DataShuffler::operator(), (arg("self"), arg("data"), arg("target")), "Populates the output matrices by randomly selecting N arrays from the input arraysets and matching targets in the most possible fair way. The 'data' and 'target' matrices will contain N rows and the number of columns that are dependent on input arraysets and target arrays.\n\n.. note::\n   We check don't 'data' and 'target' for size compatibility and is your responsibility to do so prior to calling this method.")
    ;

  class_<train::MLPRPropTrainer>("MLPRPropTrainer", "Sets an MLP to perform discrimination based on RProp: A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin Riedmiller and Heinrich Braun on IEEE International Conference on Neural Networks, pp. 586--591, 1993.", init<const mach::MLP&, size_t>((arg("machine"), arg("batch_size")), "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size. Good values for batch sizes are tens of samples. RProp is a 'batch' training algorithm. Do not try to set batch_size to a too-low value."))
    .def("reset", &train::MLPRPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero as described on the section II.C of the RProp paper.")
    .add_property("batchSize", &train::MLPRPropTrainer::getBatchSize, &train::MLPRPropTrainer::setBatchSize)
    .def("setSeed", &train::MLPRPropTrainer::setSeed, (arg("self"), arg("seed")), "Sets the seed of the internal random number generator.")
    .def("isCompatible", &train::MLPRPropTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("random", &random1, (arg("self"), arg("machine")), "Initializes a given MLP randomly, with values between [-0.1, 0.1) as advised in textbooks. You can (optionally) specify the lower and upper bound for the uniform distribution that will be used to draw values from. The default values are the ones recommended by most implementations. Be sure of what you are doing before training to change this too radically.\n\nValues are drawn using boost::uniform_real class. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("random", &random2, (arg("self"), arg("machine"), arg("lower_bound"), arg("upper_bound")), "Initializes a given MLP randomly. You can (optionally) specify the lower and upper bound for the uniform distribution that will be used to draw values from. The default values are the ones recommended by most implementations. Be sure of what you are doing before training to change this too radically.\n\nValues are drawn using boost::uniform_real class. Values are taken from the range [lower_bound, upper_bound) according to the boost::random documentation.")
    .def("train", &train::MLPRPropTrainer::train, (arg("self"), arg("machine"), arg("shuffler")), "Trains the MLP to perform discrimination. The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time. Iterate as much as you want to refine the training.\n\nThe machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n\n.. note::\n   In RProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n\n.. note::\n   The machine is not initialized randomly at each train() call. It is your task to call random() once at the machine you want to train and then call train() as many times as you think are necessary. This design allows for a training criteria to be encoded outside the scope of this trainer and to this type to focus only on applying the training when requested to.")
    .def("train_", &train::MLPRPropTrainer::train_, (arg("self"), arg("machine"), arg("shuffler")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
    .def("__test__", &train::MLPRPropTrainer::__test__, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above which receives a single input blitz::Array<double,2> and a target and trains the network using this. Note this method is not useful for real training sessions, only for testing the code. Avoid using this!")
    ;
}
